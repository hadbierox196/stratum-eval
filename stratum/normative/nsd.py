from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StakeholderRecord(BaseModel):
    name: str
    role: str
    affiliation: str
    consultation_date: date
    consultation_method: str  # e.g. "interview", "survey", "focus_group"


class ExclusionRecord(BaseModel):
    group_description: str
    reason_for_exclusion: str
    mitigation_plan: str  # what will be done to address this gap


class NormativeSpecificationDocument(BaseModel):
    use_case: str = Field(..., description="Plain-language description of what the model decides or informs.")

    fairness_criterion: Literal["equalized_odds", "calibration", "demographic_parity"] = Field(
        ...,
        description=(
            "The chosen fairness criterion. Because Chouldechova's theorem proves calibration and "
            "equalized odds cannot simultaneously hold when base rates differ across groups, exactly "
            "one criterion must be selected and justified. This is a normative choice, not a technical one."
        ),
    )

    fairness_rationale: str = Field(
        ...,
        description=(
            "Why this criterion was chosen. Must explicitly acknowledge what the choice costs "
            "groups not favored by it. A rationale that does not name the trade-off will be rejected."
        ),
    )

    acceptable_fnr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable false-negative rate across all evaluated groups.",
    )

    acceptable_fpr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable false-positive rate across all evaluated groups.",
    )

    stakeholders_represented: list[StakeholderRecord] = Field(
        ...,
        min_length=1,
        description="Every stakeholder group consulted. At least one record is required.",
    )

    stakeholders_not_represented: list[ExclusionRecord] = Field(
        ...,
        description=(
            "Groups that were NOT consulted. This field is required even if empty — "
            "an empty list is an explicit claim that no relevant group was excluded."
        ),
    )

    monitoring_frequency: str = Field(
        ...,
        description=(
            "How often performance will be re-evaluated in deployment (e.g. 'monthly', "
            "'every 10,000 predictions', 'on each dataset version bump')."
        ),
    )

    validity_horizon: date = Field(
        ...,
        description=(
            "Date after which this NSD must be reviewed and re-signed. "
            "The evaluation will not run past this date without a renewed NSD."
        ),
    )

    sunset_condition: str = Field(
        ...,
        description=(
            "The condition under which the model must be withdrawn from deployment "
            "regardless of the validity_horizon (e.g. 'FPR exceeds 0.15 in any group for two consecutive months')."
        ),
    )

    @model_validator(mode="after")
    def rationale_must_acknowledge_tradeoff(self) -> NormativeSpecificationDocument:
        tradeoff_keywords = [
            "cost", "trade", "sacrifice", "penaliz", "disadvantag",
            "harm", "burden", "inequit", "disproportion", "miss",
        ]
        rationale_lower = self.fairness_rationale.lower()
        if not any(kw in rationale_lower for kw in tradeoff_keywords):
            raise ValueError(
                "fairness_rationale must explicitly acknowledge the costs borne by groups "
                "not favored by the chosen criterion. Add language describing the trade-off "
                "(e.g. what equalized_odds costs in calibration terms, or vice versa)."
            )
        return self

    @model_validator(mode="after")
    def validity_horizon_must_be_future(self) -> NormativeSpecificationDocument:
        if self.validity_horizon <= date.today():
            raise ValueError(
                f"validity_horizon ({self.validity_horizon}) must be a future date. "
                "An expired NSD means the evaluation context has not been re-examined."
            )
        return self
