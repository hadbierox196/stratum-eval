from __future__ import annotations

import yaml
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from stratum.normative.nsd import NormativeSpecificationDocument


class StratumNSDError(Exception):
    """
    Raised when an NSD file is missing, structurally invalid, or fails
    normative checks. This is always a hard error — the evaluation will not run.
    """

    def __init__(self, message: str, field_errors: list[dict] | None = None):
        self.field_errors = field_errors or []
        detail = self._format(message, self.field_errors)
        super().__init__(detail)

    @staticmethod
    def _format(message: str, field_errors: list[dict]) -> str:
        lines = [f"[StratumNSDError] {message}"]
        if field_errors:
            lines.append("\nField-level errors:")
            for err in field_errors:
                loc = " -> ".join(str(l) for l in err.get("loc", []))
                msg = err.get("msg", "")
                lines.append(f"  • {loc}: {msg}")
        lines.append(
            "\nThe evaluation cannot run until the NSD is valid. "
            "Normative decisions cannot be defaulted or inferred — they must be made explicitly."
        )
        return "\n".join(lines)


def load_nsd(path: str | Path) -> NormativeSpecificationDocument:
    """
    Load and validate an NSD from a YAML file.
    Any validation failure raises StratumNSDError (never a warning).
    """
    path = Path(path)

    if not path.exists():
        raise StratumNSDError(
            f"NSD file not found: {path}\n"
            "Every evaluation must include a Normative Specification Document. "
            "Create one at the path specified in your evaluation config."
        )

    if path.suffix not in {".yaml", ".yml"}:
        raise StratumNSDError(
            f"NSD file must be YAML (.yaml or .yml), got: {path.suffix}"
        )

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StratumNSDError(f"NSD file is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise StratumNSDError(
            "NSD file must be a YAML mapping (key: value pairs at the top level)."
        )

    _check_expired_horizon(raw, path)

    try:
        return NormativeSpecificationDocument(**raw)
    except ValidationError as exc:
        raise StratumNSDError(
            f"NSD validation failed for: {path}",
            field_errors=exc.errors(),
        ) from exc


def _check_expired_horizon(raw: dict, path: Path) -> None:
    """
    Provide a clearer error when the validity_horizon has already passed,
    before Pydantic parsing obscures it.
    """
    horizon_raw = raw.get("validity_horizon")
    if horizon_raw is None:
        return  # Pydantic will catch the missing field
    try:
        horizon = date.fromisoformat(str(horizon_raw))
    except ValueError:
        return  # Pydantic will catch the bad format
    if horizon <= date.today():
        raise StratumNSDError(
            f"NSD has expired (validity_horizon: {horizon}). "
            f"The normative context documented in {path.name} has not been re-examined "
            "since that date. Update and re-sign the NSD before re-running the evaluation."
        )


def validate_nsd_dict(data: dict) -> NormativeSpecificationDocument:
    """Validate an already-parsed dict (useful in tests or programmatic flows)."""
    try:
        return NormativeSpecificationDocument(**data)
    except ValidationError as exc:
        raise StratumNSDError(
            "NSD validation failed (from dict)",
            field_errors=exc.errors(),
        ) from exc
