"""Weight-perturbation fragility (white-box models only).

For each of K noisy copies of the network, every Linear layer's weight (and bias) tensor W
receives i.i.d. Gaussian noise with per-entry std ``tau * ||W||_F / sqrt(numel(W))``, so the
noise tensor's norm is about ``tau * ||W||_F``. sigma(x) is the per-query standard deviation
of the K perturbed predictions.

This is an empirical sensitivity diagnostic, not a certified bound.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .surrogate import is_white_box


class BlackBoxModelError(ValueError):
    """Raised when fragility is requested for a model without accessible weights."""


def perturb_network(net: nn.Module, tau: float, generator: torch.Generator, include_bias: bool = True) -> nn.Module:
    """Return a deep copy of ``net`` with layer-norm-scaled Gaussian noise added to each Linear layer."""
    noisy = copy.deepcopy(net)
    with torch.no_grad():
        for module in noisy.modules():
            if not isinstance(module, nn.Linear):
                continue
            params = [module.weight] + ([module.bias] if include_bias and module.bias is not None else [])
            for p in params:
                scale = tau * float(p.norm()) / math.sqrt(p.numel())
                p.add_(torch.randn(p.shape, generator=generator) * scale)
    return noisy


def perturbation_sigma(
    model: object, X: np.ndarray, K: int = 30, tau: float = 0.02, seed: int = 0
) -> np.ndarray:
    """Per-query std of predictions over ``K`` weight-perturbed copies of a white-box torch MLP."""
    if not is_white_box(model):
        raise BlackBoxModelError("fragility requires a white-box torch model (weights not available)")
    gen = torch.Generator().manual_seed(seed)
    preds = np.stack(
        [model.predict_with_net(perturb_network(model.net, tau, gen), X) for _ in range(K)]  # type: ignore[attr-defined]
    )
    return preds.std(axis=0)
