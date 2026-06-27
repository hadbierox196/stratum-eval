import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvalDataset:
    df: pd.DataFrame
    label_col: str
    score_col: str
    group_col: str
    time_col: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        required = [self.label_col, self.score_col, self.group_col]
        if self.time_col:
            required.append(self.time_col)
        for col in required:
            if col not in self.df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(self.df.columns)}"
                )

    def __repr__(self):
        n = len(self.df)
        prev = self.df[self.label_col].mean()
        return f"EvalDataset(n={n}, prevalence={prev:.3f}, group_col='{self.group_col}')"
