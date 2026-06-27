from stratum.normative.nsd import NormativeSpec
from stratum.exceptions import StratumValidationError

def validate_nsd_dict(d: dict) -> NormativeSpec:
    try:
        return NormativeSpec(**d)
    except Exception as e:
        raise StratumValidationError(str(e)) from e
