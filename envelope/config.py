"""Configuration loading with defaults (YAML)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_FALLBACK: dict[str, Any] = {
    "seed": 0,
    "alpha": 0.1,
    "interval_method": "split",
    "envelope": {"k": 5, "percentile": 95, "pvalue_beta": 0.05, "flag": "percentile"},
    "gate": {"tolerance": 0.6, "escalate_score": 1.5},
    "coverage": {"ci_level": 0.95, "delta": 0.1, "small_sample_correction": False},
    "group_conditional": {"n_bins": 4, "min_cal_per_bin": 30},
    "error_tolerance": 1.0,
    "surrogate": {"mlp_hidden": [64, 64], "mlp_epochs": 1500, "mlp_lr": 0.01},
    "fragility": {"K": 30, "tau": 0.02, "beta": None},
    "tilt": {
        "risk_to_tilt": {"low": 0.0, "medium": 1.0, "high": 3.0},
        "tail_fraction": 0.1,
        "worst_region_bins": 4,
        "min_bin_count": 5,
        "demo_tilts": [0.0, 2.0, -2.0],
        "demo_corrupt_fraction": 0.05,
    },
    "adequacy": {
        "low": {"max_coverage_gap": 0.10, "max_pct_outside": 30.0, "min_r2": 0.70},
        "medium": {"max_coverage_gap": 0.05, "max_pct_outside": 15.0, "min_r2": 0.85},
        "high": {"max_coverage_gap": 0.02, "max_pct_outside": 5.0, "min_r2": 0.95},
    },
    "units": {},
    "bounds": {},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML config (falling back to built-in defaults) and apply overrides."""
    cfg = copy.deepcopy(_FALLBACK)
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        with open(p) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
