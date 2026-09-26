"""Tilted empirical risk minimisation (TERM).

L_t = (1/t) * log(mean(exp(t * loss_i)))

* t -> 0 recovers the mean loss (ERM);
* t > 0 up-weights the largest losses (approaches the max; can amplify outliers);
* t < 0 down-weights the largest losses (approaches the min; robust to outliers).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

_T_EPS = 1e-8


def tilted_loss(losses: torch.Tensor, t: float) -> torch.Tensor:
    """Numerically stable tilted aggregate of per-sample losses (falls back to the mean at t=0)."""
    if abs(t) < _T_EPS:
        return losses.mean()
    n = losses.numel()
    # float64 avoids cancellation in (logsumexp - log n) / t when |t| is small
    z = t * losses.reshape(-1).double()
    return ((torch.logsumexp(z, dim=0) - math.log(n)) / t).to(losses.dtype)


def tilted_risk_np(losses: np.ndarray, t: float) -> float:
    """NumPy version of :func:`tilted_loss` for evaluation metrics."""
    losses = np.asarray(losses, dtype=float).ravel()
    if abs(t) < _T_EPS:
        return float(losses.mean())
    z = t * losses
    m = z.max()
    return float((m + np.log(np.mean(np.exp(z - m)))) / t)


def train_surrogate(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    tilt: float = 0.0,
    epochs: int = 1500,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> list[float]:
    """Train ``model`` in place with the tilted squared-error objective (full-batch Adam).

    ``X`` and ``y`` should already be standardised; the per-sample loss is the squared error
    in standardised target units, so ``tilt`` has a scale-free meaning. Returns the loss history.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt).reshape(-1)
        losses = (pred - yt) ** 2
        obj = tilted_loss(losses, tilt)
        obj.backward()
        opt.step()
        sched.step()
        history.append(float(obj.detach()))
    model.eval()
    return history
