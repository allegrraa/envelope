"""Surrogate models: a black-box sklearn gradient-boosting model and a white-box torch MLP."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn

from .term import train_surrogate


class Surrogate(Protocol):
    """Minimal surrogate interface."""

    white_box: bool
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class GBRSurrogate:
    """Gradient-boosting surrogate. Treated as black-box (no torch weights to perturb)."""

    white_box = False
    name = "GradientBoostingRegressor"

    def __init__(self, seed: int = 0) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBRSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=float))


def build_mlp(n_in: int, hidden: Sequence[int] = (64, 64)) -> nn.Sequential:
    """Small SiLU MLP with a scalar output."""
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class TorchMLPSurrogate:
    """White-box MLP surrogate (CPU). Inputs/targets are standardised outside the network.

    Parameters
    ----------
    tilt: tilt ``t`` of the TERM objective used for training (0 = ordinary MSE).
    """

    white_box = True

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        tilt: float = 0.0,
        epochs: int = 1500,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> None:
        self.hidden = tuple(hidden)
        self.tilt = float(tilt)
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net: nn.Sequential | None = None
        self.history: list[float] = []

    @property
    def name(self) -> str:
        return f"TorchMLP{list(self.hidden)} (tilt t={self.tilt:g})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-12
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-12)
        torch.manual_seed(self.seed)
        self.net = build_mlp(X.shape[1], self.hidden)
        self.history = train_surrogate(
            self.net,
            (X - self.x_mean) / self.x_std,
            (y - self.y_mean) / self.y_std,
            tilt=self.tilt,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
        )
        return self

    def predict_with_net(self, net: nn.Module, X: np.ndarray) -> np.ndarray:
        """Predict with an arbitrary network sharing this surrogate's standardisation."""
        Z = torch.as_tensor((np.asarray(X, dtype=float) - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            out = net(Z).reshape(-1).numpy().astype(float)
        return out * self.y_std + self.y_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        return self.predict_with_net(self.net, X)


def is_white_box(model: object) -> bool:
    """True when the model exposes torch weights that can be perturbed."""
    return bool(getattr(model, "white_box", False)) and isinstance(getattr(model, "net", None), nn.Module)


def make_surrogate(kind: str, cfg: dict, tilt: float = 0.0, seed: int = 0) -> Surrogate:
    """Factory: ``kind`` is ``"gbr"`` or ``"mlp"``."""
    if kind == "gbr":
        return GBRSurrogate(seed=seed)
    if kind == "mlp":
        s = cfg.get("surrogate", {})
        return TorchMLPSurrogate(
            hidden=s.get("mlp_hidden", (64, 64)),
            tilt=tilt,
            epochs=int(s.get("mlp_epochs", 1500)),
            lr=float(s.get("mlp_lr", 1e-2)),
            seed=seed,
        )
    raise ValueError(f"unknown surrogate kind: {kind}")


def _fingerprint(model: TorchMLPSurrogate, X: np.ndarray, y: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(repr((model.hidden, model.tilt, model.epochs, model.lr, model.seed)).encode())
    h.update(np.ascontiguousarray(X, dtype=float).tobytes())
    h.update(np.ascontiguousarray(y, dtype=float).tobytes())
    return h.hexdigest()


def save_mlp(model: TorchMLPSurrogate, path: str, X: np.ndarray, y: np.ndarray) -> None:
    """Save weights + standardisation, tagged with a fingerprint of the settings and training data."""
    assert model.net is not None
    torch.save({
        "state_dict": model.net.state_dict(),
        "x_mean": torch.as_tensor(model.x_mean), "x_std": torch.as_tensor(model.x_std),
        "y_mean": model.y_mean, "y_std": model.y_std,
        "fingerprint": _fingerprint(model, X, y),
    }, path)


def load_or_train_mlp(cfg: dict, X: np.ndarray, y: np.ndarray, path: str | None = None, seed: int = 0) -> TorchMLPSurrogate:
    """Load saved weights if they were trained with the same settings and data, else train (and save)."""
    import os

    model = make_surrogate("mlp", cfg, seed=seed)
    assert isinstance(model, TorchMLPSurrogate)
    if path and os.path.exists(path):
        try:
            blob = torch.load(path, weights_only=True)
            if blob["fingerprint"] == _fingerprint(model, X, y):
                model.net = build_mlp(X.shape[1], model.hidden)
                model.net.load_state_dict(blob["state_dict"])
                model.net.eval()
                model.x_mean, model.x_std = blob["x_mean"].numpy(), blob["x_std"].numpy()
                model.y_mean, model.y_std = float(blob["y_mean"]), float(blob["y_std"])
                return model
        except (OSError, KeyError, RuntimeError):
            pass
    model.fit(X, y)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_mlp(model, path, X, y)
    return model
