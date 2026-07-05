"""
MC-Dropout + Deep Ensemble utilities for uncertainty estimation.
"""
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropoutMLP(nn.Module):
    """MLP with dropout applied at inference time for MC-Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    def predict_mc(self, x: torch.Tensor, n_samples: int = 30) -> np.ndarray:
        """
        Enable dropout at inference, sample n_samples times.
        Returns array of shape (n_samples, N).
        """
        self.train()  # enables dropout
        with torch.no_grad():
            samples = [self.forward(x).squeeze(-1).cpu().numpy() for _ in range(n_samples)]
        self.eval()
        return np.stack(samples, axis=0)  # (n_samples, N)


def build_ensemble(
    input_dim: int,
    n_members: int = 5,
    hidden_dim: int = 64,
    dropout_p: float = 0.2,
) -> List[MCDropoutMLP]:
    """
    Build a deep ensemble: list of independently initialized MLPs.
    Each member trained separately from a different random seed.
    """
    return [MCDropoutMLP(input_dim, hidden_dim, dropout_p) for _ in range(n_members)]


def train_ensemble_member(
    model: MCDropoutMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
) -> MCDropoutMLP:
    """Train a single ensemble member with binary cross-entropy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X).squeeze(-1)
        loss = F.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def ensemble_predict_proba(
    members: List[MCDropoutMLP],
    X: np.ndarray,
    n_mc_samples: int = 30,
) -> np.ndarray:
    """
    Collect predictions from all ensemble members × MC-Dropout samples.
    Returns array of shape (n_members * n_mc_samples, N).
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    all_preds = []
    for member in members:
        mc_preds = member.predict_mc(X_t, n_samples=n_mc_samples)  # (n_mc, N)
        all_preds.append(mc_preds)
    return np.concatenate(all_preds, axis=0)  # (n_members * n_mc, N)
