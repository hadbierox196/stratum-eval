import numpy as np
import pandas as pd

class SyntheticConnector:
    """
    Generates a synthetic binary-outcome dataset with two groups
    whose base rates differ by at least 0.10 (required for Figure 3).
    """
    def __init__(self, n: int = 800, seed: int = 42):
        self.n = n
        self.seed = seed

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n_a = self.n // 2
        n_b = self.n - n_a

        # Group A: prevalence ~0.30
        feat_a = rng.normal(loc=[0.0, 1.0, -0.5], scale=1.0, size=(n_a, 3))
        logit_a = 0.5 * feat_a[:, 0] + 0.3 * feat_a[:, 1] - 0.2 * feat_a[:, 2] - 0.8
        prob_a  = 1 / (1 + np.exp(-logit_a))
        y_a     = rng.binomial(1, prob_a)

        # Group B: prevalence ~0.45 (>0.10 gap)
        feat_b = rng.normal(loc=[0.5, 0.5, 0.0], scale=1.0, size=(n_b, 3))
        logit_b = 0.5 * feat_b[:, 0] + 0.3 * feat_b[:, 1] - 0.2 * feat_b[:, 2] - 0.2
        prob_b  = 1 / (1 + np.exp(-logit_b))
        y_b     = rng.binomial(1, prob_b)

        feats = np.vstack([feat_a, feat_b])
        scores = np.concatenate([prob_a, prob_b])
        outcomes = np.concatenate([y_a, y_b])
        groups = np.array(["A"] * n_a + ["B"] * n_b)
        time_idx = np.arange(self.n)

        df = pd.DataFrame(feats, columns=["feature_1", "feature_2", "feature_3"])
        df["score"]   = scores
        df["outcome"] = outcomes
        df["group"]   = groups
        df["time_idx"] = time_idx
        return df
