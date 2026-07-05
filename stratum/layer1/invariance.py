"""
IRM (linear penalty) and v-REx invariance penalties.

Both return a scalar torch.Tensor so they can be added to a training loss
or evaluated post-hoc on frozen representations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def irm_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Linear IRM penalty (Arjovsky et al. 2019, eq. 4).

    Computes the gradient norm of a fixed scalar classifier (w=1) applied to
    the per-environment loss.  A large value means the optimal linear head
    differs across environments — i.e. the representation is not invariant.

    Args:
        logits: shape (n,) or (n, 1) — raw model outputs for one environment.
        labels: shape (n,) — binary {0, 1} ground-truth labels.

    Returns:
        Scalar tensor.  Add across environments and multiply by lambda.
    """
    logits = logits.view(-1, 1).float()
    labels = labels.view(-1, 1).float()

    # Fixed "dummy" scale parameter — gradient w.r.t. this is the penalty
    scale = torch.ones(1, requires_grad=True, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits * scale, labels)
    grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return grad.pow(2)


def vrex_penalty(env_losses: list[Tensor]) -> Tensor:
    """
    v-REx penalty (Krueger et al. 2021).

    Variance of per-environment mean losses.  Tractable for nonlinear models
    because it requires no second-order gradients — only the scalar mean loss
    per environment.

    Args:
        env_losses: list of scalar tensors, one mean loss per environment.
                    Must have len >= 2 (caller is responsible for this check).

    Returns:
        Scalar tensor — variance across environments.
    """
    losses = torch.stack(env_losses)          # shape (E,)
    return losses.var()                        # unbiased by default


def irm_penalty_multiclass(logits: Tensor, labels: Tensor) -> Tensor:
    """
    IRM penalty for K-class problems (cross-entropy variant).

    Args:
        logits: shape (n, K).
        labels: shape (n,) with integer class indices.

    Returns:
        Scalar tensor.
    """
    logits = logits.float()
    labels = labels.long()

    scale = torch.ones(1, requires_grad=True, device=logits.device)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return grad.pow(2)
