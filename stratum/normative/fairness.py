from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


@dataclass
class GroupStats:
    name: str
    base_rate: float        # prevalence in this group
    predicted_positive_rate: float  # P(score >= threshold)


def chouldechova_fpr_from_fnr(
    fnr: float,
    base_rate: float,
    predicted_positive_rate: float,
) -> float:
    """
    Chouldechova (2017) impossibility relation, solved for FPR:

        PPV = TP / (TP + FP)
        Constraint: if PPV is fixed across groups but base rates differ,
        then FNR and FPR cannot both be equal across groups.

    The bound that links FNR, FPR, base_rate (p), and predicted positive
    rate (p_hat) is:

        FPR = (p * FNR - p_hat * FNR + p_hat - p) / ((1 - p) * (1 - p_hat / p_hat))

    We use the direct form derived from the confusion matrix identities:

        p_hat = p * (1 - FNR) + (1 - p) * FPR
        =>  FPR = (p_hat - p * (1 - FNR)) / (1 - p)

    Args:
        fnr: false-negative rate for this group
        base_rate: prevalence P(Y=1) for this group
        predicted_positive_rate: P(score >= threshold) for this group

    Returns:
        The FPR implied by the constraint (may be outside [0,1] if inputs
        are inconsistent — callers should check).
    """
    if base_rate <= 0 or base_rate >= 1:
        raise ValueError(f"base_rate must be in (0,1), got {base_rate}")
    numerator = predicted_positive_rate - base_rate * (1.0 - fnr)
    denominator = 1.0 - base_rate
    return numerator / denominator


def impossibility_fnr_fpr_frontier(
    base_rate: float,
    predicted_positive_rate: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the (FNR, FPR) frontier imposed by the impossibility constraint
    for a single group. Points where implied FPR is outside [0,1] are dropped.
    """
    fnr_grid = np.linspace(0.0, 1.0, n_points)
    fpr_values = np.array([
        chouldechova_fpr_from_fnr(fnr, base_rate, predicted_positive_rate)
        for fnr in fnr_grid
    ])
    mask = (fpr_values >= 0) & (fpr_values <= 1)
    return fnr_grid[mask], fpr_values[mask]


def plot_impossibility_bounds(
    groups: list[GroupStats],
    acceptable_fnr: float | None = None,
    acceptable_fpr: float | None = None,
    title: str = "Chouldechova Impossibility Bounds",
    figsize: tuple[float, float] = (8.0, 6.0),
) -> "plt.Figure":
    """
    Visualize the FNR-FPR frontier for each group.

    When base rates differ, the frontiers diverge — illustrating that a
    single (FNR, FPR) operating point cannot satisfy equalized odds AND
    calibration simultaneously.

    Args:
        groups: list of GroupStats, one per demographic group
        acceptable_fnr: draw a vertical line at this FNR threshold
        acceptable_fpr: draw a horizontal line at this FPR threshold
        title: plot title
        figsize: matplotlib figure size

    Returns:
        matplotlib Figure (caller can .savefig() or .show())
    """
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(groups):
        fnr_vals, fpr_vals = impossibility_fnr_fpr_frontier(
            group.base_rate, group.predicted_positive_rate
        )
        color = colors[i % len(colors)]
        ax.plot(
            fnr_vals,
            fpr_vals,
            color=color,
            linewidth=2,
            label=f"{group.name} (base rate={group.base_rate:.2f})",
        )

    # Acceptable threshold lines
    if acceptable_fnr is not None:
        ax.axvline(
            acceptable_fnr,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Acceptable FNR = {acceptable_fnr}",
        )
    if acceptable_fpr is not None:
        ax.axhline(
            acceptable_fpr,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Acceptable FPR = {acceptable_fpr}",
        )

    # Shade the feasible region
    if acceptable_fnr is not None and acceptable_fpr is not None:
        ax.fill_betweenx(
            [0, acceptable_fpr],
            0,
            acceptable_fnr,
            alpha=0.08,
            color="green",
            label="Jointly acceptable region",
        )

    ax.set_xlabel("False-Negative Rate (FNR)", fontsize=12)
    ax.set_ylabel("False-Positive Rate (FPR)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    n_groups = len(groups)
    if n_groups >= 2:
        _annotate_divergence(ax, groups)

    return fig


def _annotate_divergence(ax: "plt.Axes", groups: list[GroupStats]) -> None:
    """Add an annotation explaining why frontiers diverge."""
    base_rates = [g.base_rate for g in groups]
    br_range = max(base_rates) - min(base_rates)
    annotation = (
        f"Base-rate spread: {br_range:.2f}\n"
        "Chouldechova (2017): when base rates differ,\n"
        "calibration and equalized odds cannot both hold."
    )
    ax.text(
        0.98, 0.02, annotation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )


def summarize_impossibility(groups: list[GroupStats]) -> str:
    """
    Return a plain-text summary of the impossibility tension across groups.
    Useful for NSD generation and audit logs.
    """
    lines = ["Chouldechova Impossibility Summary", "=" * 40]
    base_rates = [g.base_rate for g in groups]
    if max(base_rates) - min(base_rates) < 1e-6:
        lines.append(
            "Base rates are effectively equal across groups. "
            "The impossibility constraint is not binding — calibration and "
            "equalized odds can be simultaneously satisfied."
        )
        return "\n".join(lines)

    lines.append(
        "Base rates differ across groups. By Chouldechova's theorem, "
        "calibration and equalized odds CANNOT both hold.\n"
    )
    for g in groups:
        lines.append(f"  Group: {g.name}")
        lines.append(f"    Base rate (prevalence): {g.base_rate:.4f}")
        lines.append(f"    Predicted positive rate: {g.predicted_positive_rate:.4f}")
        # Show the FNR=0 implied FPR (best-case FNR scenario)
        try:
            fpr_at_zero_fnr = chouldechova_fpr_from_fnr(0.0, g.base_rate, g.predicted_positive_rate)
            lines.append(f"    FPR if FNR=0 (lower bound on FPR): {fpr_at_zero_fnr:.4f}")
        except ValueError:
            pass
        lines.append("")

    lines.append(
        "Implication: your NSD must choose either calibration or equalized odds "
        "and document what the unchosen criterion costs the groups it would have protected."
    )
    return "\n".join(lines)
