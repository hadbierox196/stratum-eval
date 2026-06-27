from pydantic import BaseModel
from datetime import date
from typing import Literal, Optional

class StakeholderRecord(BaseModel):
    name: str
    role: str
    consultation_date: Optional[str] = None
    consultation_method: Optional[str] = None
    exclusion_reason: Optional[str] = None

class NormativeSpec(BaseModel):
    use_case: str
    deployment_context: str
    fairness_criterion: Literal["equalized_odds", "calibration", "demographic_parity"]
    fairness_rationale: str
    acceptable_fnr: float
    acceptable_fpr: float
    stakeholders_represented: list[StakeholderRecord] = []
    stakeholders_not_represented: list[StakeholderRecord] = []
    monitoring_frequency: str = "quarterly"
    validity_horizon: date
    sunset_condition: str
