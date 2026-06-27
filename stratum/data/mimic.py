import pandas as pd

class MIMICIVConnector:
    """Placeholder — requires local MIMIC-IV access."""
    def load(self) -> pd.DataFrame:
        raise NotImplementedError(
            "MIMICIVConnector requires credentialed MIMIC-IV data. "
            "See https://physionet.org/content/mimiciv/"
        )
