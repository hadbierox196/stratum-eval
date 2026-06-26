from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SyntheticSepsisConfig:
    """
    Controls the synthetic sepsis cohort generator.

    Designed to reproduce the key statistical properties relevant to
    Chouldechova's impossibility theorem: differing base rates across
    demographic groups, realistic SOFA distributions, and correlated outcomes.
    """
    n_patients: int = 2000
    random_seed: int = 42

    # Group definitions: (label, prevalence_fraction, sepsis_base_rate)
    # Fractions must sum to 1.0
    groups: list[tuple[str, float, float]] = field(default_factory=lambda: [
        ("White",          0.55, 0.28),
        ("Black",          0.20, 0.34),   # higher base rate — key for impossibility demo
        ("Hispanic",       0.15, 0.30),
        ("Asian",          0.07, 0.25),
        ("Other/Unknown",  0.03, 0.31),
    ])

    sofa_mean: float = 6.2
    sofa_std: float = 3.1
    sofa_min: int = 0
    sofa_max: int = 24

    # Noise added to logits to simulate prediction model imperfection
    model_noise_std: float = 1.2

    # Feature correlations (simplified)
    age_mean: float = 62.0
    age_std: float = 16.0
    age_sepsis_or: float = 1.02  # odds ratio per year of age on sepsis


def build_synthetic_sepsis_cohort(
    config: SyntheticSepsisConfig | None = None,
) -> pd.DataFrame:
    """
    Generate a fully synthetic MIMIC-IV-like sepsis cohort.

    Returns a DataFrame with the same schema as
    stratum.connectors.mimic.build_sepsis_cohort(), allowing tests and CI
    pipelines to run without PhysioNet credentials.

    The synthetic data is NOT suitable for clinical research — it is designed
    only to exercise stratum-eval's evaluation pipeline.
    """
    cfg = config or SyntheticSepsisConfig()
    rng = np.random.default_rng(cfg.random_seed)

    _validate_group_fractions(cfg.groups)

    rows = []
    subject_id = 1

    for group_label, group_frac, sepsis_rate in cfg.groups:
        n = round(cfg.n_patients * group_frac)
        rows.extend(
            _generate_group(rng, group_label, sepsis_rate, n, cfg, subject_id)
        )
        subject_id += n

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
    return df


def _validate_group_fractions(groups: list[tuple[str, float, float]]) -> None:
    total = sum(frac for _, frac, _ in groups)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"SyntheticSepsisConfig.groups fractions must sum to 1.0, got {total:.6f}"
        )


def _generate_group(
    rng: np.random.Generator,
    group_label: str,
    sepsis_rate: float,
    n: int,
    cfg: SyntheticSepsisConfig,
    subject_id_start: int,
) -> list[dict]:
    rows = []

    ages = rng.normal(cfg.age_mean, cfg.age_std, n).clip(18, 99)

    # Adjust sepsis probability by age (logistic)
    log_odds_base = np.log(sepsis_rate / (1 - sepsis_rate))
    log_odds = log_odds_base + (ages - cfg.age_mean) * np.log(cfg.age_sepsis_or)
    p_sepsis = 1.0 / (1.0 + np.exp(-log_odds))

    sepsis_labels = rng.binomial(1, p_sepsis).astype(bool)

    sofa_scores = rng.normal(cfg.sofa_mean, cfg.sofa_std, n)
    sofa_scores = np.clip(sofa_scores, cfg.sofa_min, cfg.sofa_max).round().astype(int)

    # Synthetic model scores (noisy logits → probabilities)
    true_logits = 0.4 * (sofa_scores - cfg.sofa_mean) + sepsis_labels.astype(float) * 1.5
    noisy_logits = true_logits + rng.normal(0, cfg.model_noise_std, n)
    model_scores = 1.0 / (1.0 + np.exp(-noisy_logits))

    genders = rng.choice(["M", "F"], n, p=[0.52, 0.48])

    # Mortality: ~20% in sepsis, ~3% without
    p_mort = np.where(sepsis_labels, 0.20 + 0.008 * (sofa_scores - 6).clip(0), 0.03)
    mortality = rng.binomial(1, p_mort.clip(0, 1)).astype(bool)

    for i in range(n):
        rows.append({
            "subject_id":            subject_id_start + i,
            "hadm_id":               (subject_id_start + i) * 10,
            "stay_id":               (subject_id_start + i) * 100,
            "age":                   float(ages[i]),
            "gender":                genders[i],
            "race":                  group_label,
            "sofa_score":            int(sofa_scores[i]),
            "sepsis_label":          bool(sepsis_labels[i]),
            "model_score":           float(model_scores[i]),
            "mortality_hospital":    bool(mortality[i]),
            "sepsis_onset_offset_hr": float(rng.exponential(8.0)),
            "_synthetic":            True,   # flag for downstream checks
        })

    return rows


def describe_cohort(df: pd.DataFrame) -> str:
    """Return a plain-text summary of a synthetic or real cohort."""
    lines = ["Cohort Summary", "=" * 40]
    lines.append(f"Total patients: {len(df):,}")

    if "sepsis_label" in df.columns:
        overall_rate = df["sepsis_label"].mean()
        lines.append(f"Overall sepsis rate: {overall_rate:.3f}")

    if "race" in df.columns:
        lines.append("\nBase rates by group:")
        for group, grp_df in df.groupby("race"):
            if "sepsis_label" in grp_df.columns:
                rate = grp_df["sepsis_label"].mean()
                lines.append(f"  {group:<20} n={len(grp_df):>5,}  sepsis rate={rate:.3f}")

    if df.get("_synthetic", pd.Series([False])).any():
        lines.append("\n⚠ This is SYNTHETIC data. Do not use for clinical research.")

    return "\n".join(lines)
