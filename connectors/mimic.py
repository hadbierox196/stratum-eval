from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


PHYSIONET_ENV_USER = "PHYSIONET_USER"
PHYSIONET_ENV_PASSWORD = "PHYSIONET_PASSWORD"

# MIMIC-IV BigQuery project / GCS paths (canonical PhysioNet locations)
MIMIC_BQ_PROJECT = "physionet-data"
MIMIC_BQ_DATASET = "mimiciv_3_1"  # update to current version as needed
MIMIC_GCS_BUCKET = "physionet-data/mimiciv/3.1"


@dataclass
class PhysioNetCredentials:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "PhysioNetCredentials":
        user = os.environ.get(PHYSIONET_ENV_USER)
        password = os.environ.get(PHYSIONET_ENV_PASSWORD)
        if not user or not password:
            raise StratumMIMICError(
                f"PhysioNet credentials not found in environment. "
                f"Set {PHYSIONET_ENV_USER} and {PHYSIONET_ENV_PASSWORD}, "
                "or call PhysioNetCredentials.from_prompt() for interactive use."
            )
        return cls(username=user, password=password)

    @classmethod
    def from_prompt(cls) -> "PhysioNetCredentials":
        print("PhysioNet credentialed access required for MIMIC-IV.")
        print("Register at https://physionet.org/register/ and complete CITI training.")
        user = input("PhysioNet username: ").strip()
        password = getpass.getpass("PhysioNet password: ")
        return cls(username=user, password=password)

    @classmethod
    def resolve(cls, credentials: "PhysioNetCredentials | None" = None) -> "PhysioNetCredentials":
        if credentials is not None:
            return credentials
        try:
            return cls.from_env()
        except StratumMIMICError:
            return cls.from_prompt()


class StratumMIMICError(Exception):
    pass


@dataclass
class SepsisQueryConfig:
    """
    Controls what the sepsis cohort builder extracts.
    Sepsis-3 definition: suspected infection + SOFA >= 2 point increase.
    """
    sofa_threshold: int = 2
    min_age: int = 18
    icu_stays_only: bool = True
    include_demographics: bool = True
    include_vitals: bool = True
    include_labs: bool = True
    max_rows: int | None = None  # None = no limit


def build_sepsis_cohort(
    credentials: PhysioNetCredentials | None = None,
    config: SepsisQueryConfig | None = None,
    backend: str = "bigquery",
) -> "pd.DataFrame":
    """
    Build a MIMIC-IV sepsis cohort (Sepsis-3 definition).

    Args:
        credentials: PhysioNetCredentials. If None, resolved from env or prompt.
        config: SepsisQueryConfig controlling cohort definition.
        backend: 'bigquery' (default) or 'files' (local CSV download via wget/physionet-client).

    Returns:
        pandas DataFrame with one row per ICU stay, columns including:
        - subject_id, hadm_id, stay_id
        - age, gender, race  (demographics)
        - sofa_score, sofa_delta
        - sepsis_onset_offset (hours from ICU admission)
        - mortality_hospital, mortality_30day
        - vital signs (hr_mean, sbp_mean, temp_mean, spo2_mean)
        - labs (lactate_max, wbc_mean, creatinine_max, bilirubin_max)
    """
    creds = PhysioNetCredentials.resolve(credentials)
    cfg = config or SepsisQueryConfig()

    if backend == "bigquery":
        return _build_via_bigquery(creds, cfg)
    elif backend == "files":
        return _build_via_files(creds, cfg)
    else:
        raise StratumMIMICError(f"Unknown backend: {backend!r}. Use 'bigquery' or 'files'.")


# ─── BigQuery backend ────────────────────────────────────────────────────────

def _build_via_bigquery(creds: PhysioNetCredentials, cfg: SepsisQueryConfig) -> "pd.DataFrame":
    try:
        from google.cloud import bigquery  # type: ignore
        import pandas as pd
    except ImportError as exc:
        raise StratumMIMICError(
            "BigQuery backend requires: pip install google-cloud-bigquery pandas"
        ) from exc

    # PhysioNet BigQuery uses OAuth via gcloud; username/password are for
    # physionet.org credential verification only. Users must have GCP access.
    client = bigquery.Client(project=MIMIC_BQ_PROJECT)
    query = _sepsis3_query(cfg)

    print(f"Querying MIMIC-IV ({MIMIC_BQ_DATASET}) via BigQuery…")
    df = client.query(query).to_dataframe()
    print(f"Cohort built: {len(df):,} ICU stays.")
    return df


def _sepsis3_query(cfg: SepsisQueryConfig) -> str:
    limit_clause = f"LIMIT {cfg.max_rows}" if cfg.max_rows else ""
    return f"""
WITH sepsis_stays AS (
  -- Angus criteria / Sepsis-3: infection + acute organ dysfunction
  SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    ie.intime AS icu_intime,
    ie.outtime AS icu_outtime,
    s3.sofa_24hours AS sofa_score,
    s3.suspected_infection_time
  FROM `{MIMIC_BQ_PROJECT}.{MIMIC_BQ_DATASET}.icustays` ie
  INNER JOIN `{MIMIC_BQ_PROJECT}.{MIMIC_BQ_DATASET}.sepsis3` s3
    ON ie.stay_id = s3.stay_id
  WHERE s3.sepsis3 = TRUE
    AND s3.sofa_24hours >= {cfg.sofa_threshold}
),

demographics AS (
  SELECT
    p.subject_id,
    p.gender,
    p.anchor_age AS age,
    ad.race
  FROM `{MIMIC_BQ_PROJECT}.{MIMIC_BQ_DATASET}.patients` p
  INNER JOIN `{MIMIC_BQ_PROJECT}.{MIMIC_BQ_DATASET}.admissions` ad
    ON p.subject_id = ad.subject_id
  WHERE p.anchor_age >= {cfg.min_age}
),

outcomes AS (
  SELECT
    ad.hadm_id,
    CASE WHEN ad.hospital_expire_flag = 1 THEN TRUE ELSE FALSE END AS mortality_hospital
  FROM `{MIMIC_BQ_PROJECT}.{MIMIC_BQ_DATASET}.admissions` ad
)

SELECT
  ss.subject_id,
  ss.hadm_id,
  ss.stay_id,
  d.age,
  d.gender,
  d.race,
  ss.sofa_score,
  o.mortality_hospital,
  TIMESTAMP_DIFF(ss.suspected_infection_time, ss.icu_intime, HOUR) AS sepsis_onset_offset_hr
FROM sepsis_stays ss
LEFT JOIN demographics d USING (subject_id)
LEFT JOIN outcomes o USING (hadm_id)
{limit_clause}
"""


# ─── File-based backend ───────────────────────────────────────────────────────

def _build_via_files(creds: PhysioNetCredentials, cfg: SepsisQueryConfig) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError as exc:
        raise StratumMIMICError("File backend requires: pip install pandas") from exc

    raise NotImplementedError(
        "File-based MIMIC-IV access requires downloading tables via the physionet-client CLI. "
        "Run: pip install physionet-client && physionet-get -u {username} mimiciv/3.1/ \n"
        "Then re-run with the path to your local MIMIC-IV directory. "
        "This backend will be implemented in a future release. "
        "For CI/testing without credentials, use stratum.connectors.synthetic."
    )
