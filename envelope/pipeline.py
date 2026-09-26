"""End-to-end pipeline shared by the CLI demo, the Streamlit app and upload mode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .coverage_report import clopper_pearson, conformal_coverage_law, corrected_alpha
from .conformal import AdaptiveConformal, GroupConditionalConformal, NormalizedConformal, SplitConformal, quantile_bin_edges
from .data import (
    CheckResult,
    DemoData,
    corrupt_labels,
    make_demo_data,
    run_verification,
    simulator,
    slice_grid,
)
from .envelope import ApplicabilityDomain, benjamini_hochberg
from .fragility import perturbation_sigma
from .shift_check import shift_check
from .gate import Decision, GateConfig, gate_decisions
from .report import ReportInputs, build_report
from .surrogate import is_white_box, make_surrogate

METHODS = ["split", "adaptive", "fragility"]
METHOD_LABELS = {
    "split": "Split conformal",
    "adaptive": "Locally adaptive conformal",
    "fragility": "Fragility-normalised conformal",
}
BLACK_BOX_REASON = (
    "Weight-perturbation fragility is disabled: the model is black-box (no torch weights available). "
    "Fragility-normalised intervals and the parameter-sensitivity diagnostic require a white-box MLP."
)


def group_conditional_table(engine: "EnvelopeEngine", df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group-conditional conformal over quantile bins of the queries' envelope scores.

    Adds ``group_bin``, ``lo_group``, ``hi_group`` to ``df`` and returns a per-bin table (with the
    global split-conformal coverage for comparison) and a summary including the worst bin.
    """
    gc_cfg = cfg.get("group_conditional", {})
    k = int(gc_cfg.get("n_bins", 4))
    min_count = int(gc_cfg.get("min_cal_per_bin", 30))
    ci_level = float(cfg.get("coverage", {}).get("ci_level", 0.95))
    edges = quantile_bin_edges(df["env_score"].to_numpy(), k)
    gc = GroupConditionalConformal(engine.alpha_used["split"] or 1e-12, min_count).fit(
        engine.y_cal, engine.yhat_cal, engine.cal_env_score, edges
    )
    iv = gc.predict(df["y_pred"].to_numpy(), df["env_score"].to_numpy())
    df["group_bin"] = gc.assign(df["env_score"].to_numpy())
    df["lo_group"], df["hi_group"] = iv.lower, iv.upper
    bounds = np.r_[-np.inf, edges, np.inf]
    rows = []
    for info in gc.bins:
        b = info["bin"]
        sub = df[df["group_bin"] == b]
        row = {
            "bin": b + 1,
            "score_low": float(max(bounds[b], df["env_score"].min())),
            "score_high": float(min(bounds[b + 1], df["env_score"].max())),
            "n_query": int(len(sub)),
            "n_cal": info["n_cal"],
            "quantile_source": info["source"],
            "half_width": float(info["q"]),
        }
        if "y_true" in df and len(sub):
            hit_g = (sub["y_true"] >= sub["lo_group"]) & (sub["y_true"] <= sub["hi_group"])
            hit_s = (sub["y_true"] >= sub["lo_split"]) & (sub["y_true"] <= sub["hi_split"])
            row["coverage_group"] = float(hit_g.mean())
            row["ci_low"], row["ci_high"] = clopper_pearson(int(hit_g.sum()), len(sub), ci_level)
            row["coverage_split"] = float(hit_s.mean())
        else:
            row["coverage_group"] = row["ci_low"] = row["ci_high"] = row["coverage_split"] = float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    labelled = table["coverage_group"].notna()
    worst = table.loc[table.loc[labelled, "coverage_group"].idxmin()].to_dict() if labelled.any() else None
    info = {"n_bins": k, "min_cal_per_bin": min_count, "worst": worst,
            "n_fallback": int((table["quantile_source"] != "bin").sum())}
    return table, info


class EnvelopeEngine:
    """Fitted envelope + conformal calibrators + gate."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.alpha = float(cfg["alpha"])
        self.gate_cfg = GateConfig.from_config(cfg)
        self.normalized: NormalizedConformal | None = None

    def fit(
        self,
        X_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        yhat_cal: np.ndarray,
        sigma_cal: np.ndarray | None = None,
    ) -> "EnvelopeEngine":
        env = self.cfg["envelope"]
        seed = int(self.cfg.get("seed", 0))
        cov = self.cfg.get("coverage", {})
        n = len(y_cal)
        n_adapt = n - int(n * 0.5)  # AdaptiveConformal scores the second half of the calibration set
        self.correction = bool(cov.get("small_sample_correction", False))
        delta = float(cov.get("delta", 0.1))
        # alpha actually used by each calibrator (nominal alpha unless the small-sample correction is on)
        self.alpha_used = {
            "split": corrected_alpha(n, self.alpha, delta) if self.correction else self.alpha,
            "adaptive": corrected_alpha(n_adapt, self.alpha, delta) if self.correction else self.alpha,
            "fragility": corrected_alpha(n, self.alpha, delta) if self.correction else self.alpha,
        }
        self.n_cal = {"split": n, "adaptive": n_adapt, "fragility": n}
        self.ad = ApplicabilityDomain(k=int(env["k"]), percentile=float(env["percentile"])).fit(X_train).calibrate(X_cal)
        self.y_cal, self.yhat_cal = np.asarray(y_cal, float), np.asarray(yhat_cal, float)
        self.cal_env_score = self.ad.distance(X_cal) / self.ad.threshold
        self.flag_mode = str(env.get("flag", "percentile"))
        self.pvalue_beta = float(env.get("pvalue_beta", 0.05))
        if self.flag_mode not in {"percentile", "pvalue", "bh"}:
            raise ValueError(f"envelope.flag must be percentile, pvalue or bh (got {self.flag_mode!r})")
        self.split = SplitConformal(self.alpha_used["split"] or 1e-12).fit(y_cal, yhat_cal)
        self.adaptive = AdaptiveConformal(self.alpha_used["adaptive"] or 1e-12, seed=seed).fit(X_cal, y_cal, yhat_cal)
        if sigma_cal is not None:
            beta = self.cfg["fragility"].get("beta")
            self.normalized = NormalizedConformal(self.alpha_used["fragility"] or 1e-12, None if beta is None else float(beta)).fit(
                y_cal, yhat_cal, sigma_cal
            )
        return self

    @property
    def methods(self) -> list[str]:
        return ["split", "adaptive"] + (["fragility"] if self.normalized is not None else [])

    def evaluate(
        self,
        X: np.ndarray,
        yhat: np.ndarray,
        method: str = "split",
        sigma: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-query table: envelope, intervals for every available method, gate decision."""
        if method not in self.methods:
            raise ValueError(f"method {method!r} not available (have {self.methods})")
        X = np.asarray(X, dtype=float)
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=names)
        df["y_pred"] = yhat
        if y_true is not None:
            df["y_true"] = y_true
            df["abs_err"] = np.abs(np.asarray(y_true) - yhat)
        res = self.ad.evaluate(X)
        df["env_distance"] = res.distance
        df["env_score"] = res.score
        df["env_pvalue"] = res.pvalue
        df["env_flag_percentile"] = ~res.inside
        df["env_flag_pvalue"] = res.pvalue < self.pvalue_beta
        df["env_flag_bh"] = benjamini_hochberg(res.pvalue, self.pvalue_beta)
        df["inside"] = ~df[f"env_flag_{self.flag_mode}"].to_numpy()
        ivs = {"split": self.split.predict(yhat), "adaptive": self.adaptive.predict(X, yhat)}
        if self.normalized is not None and sigma is not None:
            df["sigma"] = sigma
            ivs["fragility"] = self.normalized.predict(yhat, sigma)
        for m, iv in ivs.items():
            df[f"lo_{m}"] = iv.lower
            df[f"hi_{m}"] = iv.upper
        df["lower"] = df[f"lo_{method}"]
        df["upper"] = df[f"hi_{method}"]
        df["half_width"] = (df["upper"] - df["lower"]) / 2.0
        df["decision"] = gate_decisions(
            df["env_score"].to_numpy(), df["half_width"].to_numpy(), self.gate_cfg,
            inside=None if self.flag_mode == "percentile" else df["inside"].to_numpy(),
        )
        if y_true is not None:
            df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
        return df


def method_table_from_labels(df: pd.DataFrame, methods: list[str], ci_level: float = 0.95) -> pd.DataFrame:
    """Empirical coverage (with Clopper-Pearson interval) and mean width per method,
    overall / inside / outside the envelope."""
    rows = []
    regions = {"overall": np.ones(len(df), bool), "inside": df["inside"].to_numpy(), "outside": ~df["inside"].to_numpy()}
    for m in methods:
        row: dict[str, Any] = {"method": m}
        for rname, mask in regions.items():
            sub = df[mask]
            row[f"coverage_{rname}"] = M.coverage(sub["y_true"], sub[f"lo_{m}"], sub[f"hi_{m}"])
            k = int(((sub["y_true"] >= sub[f"lo_{m}"]) & (sub["y_true"] <= sub[f"hi_{m}"])).sum())
            row[f"n_{rname}"] = len(sub)
            row[f"ci_low_{rname}"], row[f"ci_high_{rname}"] = clopper_pearson(k, len(sub), ci_level)
            row[f"width_{rname}"] = M.mean_width(sub[f"lo_{m}"].to_numpy(), sub[f"hi_{m}"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def method_table_from_calibration(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    sigma_cal: np.ndarray | None,
    cfg: dict[str, Any],
    n_rep: int = 20,
) -> pd.DataFrame:
    """Coverage estimate when queries have no labels: repeated 50/50 splits of the calibration set.

    This only measures in-distribution coverage; it says nothing about queries outside the envelope.
    """
    alpha = float(cfg["alpha"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    methods = ["split", "adaptive"] + (["fragility"] if sigma_cal is not None else [])
    acc: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    n = len(y_cal)
    for r in range(n_rep):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2 :]
        ivs = {
            "split": SplitConformal(alpha).fit(y_cal[a], yhat_cal[a]).predict(yhat_cal[b]),
            "adaptive": AdaptiveConformal(alpha, seed=r).fit(X_cal[a], y_cal[a], yhat_cal[a]).predict(X_cal[b], yhat_cal[b]),
        }
        if sigma_cal is not None:
            beta = cfg["fragility"].get("beta")
            nc = NormalizedConformal(alpha, None if beta is None else float(beta)).fit(y_cal[a], yhat_cal[a], sigma_cal[a])
            ivs["fragility"] = nc.predict(yhat_cal[b], sigma_cal[b])
        for m, iv in ivs.items():
            acc[m].append((M.coverage(y_cal[b], iv.lower, iv.upper), M.mean_width(iv.lower, iv.upper)))
    rows = []
    for m in methods:
        cov, wid = np.mean(acc[m], axis=0)
        rows.append({
            "method": m,
            "coverage_overall": cov, "width_overall": wid,
            "coverage_inside": float("nan"), "width_inside": float("nan"),
            "coverage_outside": float("nan"), "width_outside": float("nan"),
            "n_overall": 0, "n_inside": 0, "n_outside": 0,
            "ci_low_overall": float("nan"), "ci_high_overall": float("nan"),
            "ci_low_inside": float("nan"), "ci_high_inside": float("nan"),
            "ci_low_outside": float("nan"), "ci_high_outside": float("nan"),
        })
    return pd.DataFrame(rows)


@dataclass
class PipelineResult:
    """Everything the UI / CLI needs."""

    queries: pd.DataFrame
    engine: EnvelopeEngine
    method: str
    method_table: pd.DataFrame
    coverage_source: str
    summary: dict[str, Any]
    report_inputs: ReportInputs
    report_md: str
    fragility_available: bool
    notes: list[str] = field(default_factory=list)


def resolve_method(cfg: dict[str, Any], use_fragility: bool, white_box: bool) -> tuple[str, list[str]]:
    """Pick the interval method that drives the gate, with explanatory notes."""
    notes: list[str] = []
    method = "fragility" if use_fragility else str(cfg.get("interval_method", "split"))
    if method not in METHODS:
        notes.append(f"Unknown interval_method {method!r}; using split conformal.")
        method = "split"
    if method == "fragility" and not white_box:
        notes.append(BLACK_BOX_REASON + " Falling back to split conformal.")
        method = "split"
    return method, notes


def run_pipeline(
    *,
    feature_names: list[str],
    X_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    X_query: np.ndarray,
    yhat_query: np.ndarray,
    cfg: dict[str, Any],
    y_query: np.ndarray | None = None,
    model: object | None = None,
    risk_level: str = "medium",
    context_of_use: str = "",
    use_fragility: bool = False,
    checks: list[CheckResult] | None = None,
    reference: tuple[np.ndarray, np.ndarray, str] | None = None,
    model_name: str = "external model",
    mode: str = "upload",
    training_tilt: float | None = None,
    shift: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run envelope + conformal + gate + metrics + report on any model's outputs.

    ``shift``: a precomputed :func:`shift_check` result (it depends only on X_cal / X_query),
    computed here when omitted.
    """
    white_box = model is not None and is_white_box(model)
    method, notes = resolve_method(cfg, use_fragility, white_box)

    sigma_cal = sigma_q = None
    if white_box:
        fr = cfg["fragility"]
        seed = int(cfg.get("seed", 0))
        sigma_cal = perturbation_sigma(model, X_cal, K=int(fr["K"]), tau=float(fr["tau"]), seed=seed)
        sigma_q = perturbation_sigma(model, X_query, K=int(fr["K"]), tau=float(fr["tau"]), seed=seed)
    else:
        notes.append(BLACK_BOX_REASON)

    engine = EnvelopeEngine(cfg).fit(X_train, X_cal, y_cal, yhat_cal, sigma_cal)
    df = engine.evaluate(X_query, yhat_query, method=method, sigma=sigma_q, y_true=y_query, feature_names=feature_names)

    group_table, group_info = group_conditional_table(engine, df, cfg)

    if y_query is not None:
        mtab = method_table_from_labels(df, engine.methods, float(cfg.get("coverage", {}).get("ci_level", 0.95)))
        coverage_source = "labelled query set (includes any out-of-envelope queries)"
    else:
        mtab = method_table_from_calibration(X_cal, y_cal, yhat_cal, sigma_cal, cfg)
        coverage_source = (
            "repeated 50/50 splits of the calibration set - in-distribution only; "
            "coverage on out-of-envelope queries is NOT measured"
        )

    # ---- summary -----------------------------------------------------------------
    n = len(df)
    inside = df["inside"].to_numpy()
    far = df["env_score"].to_numpy() > engine.gate_cfg.escalate_score
    counts = {d.value: int((df["decision"] == d.value).sum()) for d in Decision}
    err_tol = float(cfg["error_tolerance"])
    summary: dict[str, Any] = {
        "n_queries": n,
        "pct_outside": 100.0 * float((~inside).mean()) if n else float("nan"),
        "pct_far_outside": 100.0 * float(far.mean()) if n else float("nan"),
        "pct_flag_percentile": 100.0 * float(df["env_flag_percentile"].mean()) if n else float("nan"),
        "pct_flag_pvalue": 100.0 * float(df["env_flag_pvalue"].mean()) if n else float("nan"),
        "pct_flag_bh": 100.0 * float(df["env_flag_bh"].mean()) if n else float("nan"),
        "gate_counts": counts,
        "pct_flagged": 100.0 * (1 - counts[Decision.TRUST.value] / n) if n else float("nan"),
        "mean_half_width": float(df["half_width"].mean()),
        "nominal_coverage": 1 - engine.alpha,
        "flag_pr": M.flag_precision_recall(~inside, df["abs_err"], err_tol) if y_query is not None else None,
        "group_table": group_table,
        "group_info": group_info,
    }
    summary["shift"] = shift if shift is not None else shift_check(X_cal, X_query, seed=int(cfg.get("seed", 0)))
    cov_cfg = cfg.get("coverage", {})
    law = conformal_coverage_law(engine.n_cal[method], engine.alpha_used[method] or 1e-12)
    summary["coverage_stats"] = {
        "ci_level": float(cov_cfg.get("ci_level", 0.95)),
        "n_cal": engine.n_cal[method],
        "law_nominal": conformal_coverage_law(engine.n_cal[method], engine.alpha),
        "law_used": law,
        "delta": float(cov_cfg.get("delta", 0.1)),
        "alpha_corrected": corrected_alpha(engine.n_cal[method], engine.alpha, float(cov_cfg.get("delta", 0.1))),
        "correction_applied": engine.correction,
        "alpha_used": engine.alpha_used[method],
    }

    # ---- validation ----------------------------------------------------------------
    if reference is not None:
        y_ref, yhat_ref, ref_label = reference
    else:
        y_ref, yhat_ref, ref_label = y_cal, yhat_cal, "calibration set (no separate reference data provided)"
    validation: dict[str, Any] = {
        "reference_label": ref_label,
        "n": int(len(y_ref)),
        "rmse": M.rmse(y_ref, yhat_ref),
        "r2": M.r2(y_ref, yhat_ref),
    }
    if y_query is not None:
        for name, mask in (("query_inside", inside), ("query_outside", ~inside)):
            if mask.sum() >= 2:
                validation[name] = {
                    "n": int(mask.sum()),
                    "rmse": M.rmse(df["y_true"][mask], df["y_pred"][mask]),
                    "r2": M.r2(df["y_true"][mask].to_numpy(), df["y_pred"][mask].to_numpy()),
                }

    # ---- fragility diagnostic --------------------------------------------------------
    if white_box and sigma_q is not None:
        s_in = float(np.mean(sigma_q[inside])) if inside.any() else float("nan")
        s_out = float(np.mean(sigma_q[~inside])) if (~inside).any() else float("nan")
        fragility = {
            "available": True,
            "K": int(cfg["fragility"]["K"]),
            "tau": float(cfg["fragility"]["tau"]),
            "beta": float(engine.normalized.beta) if engine.normalized else None,
            "mean_sigma_inside": s_in,
            "mean_sigma_outside": s_out,
            "used_for_intervals": method == "fragility",
        }
    else:
        fragility = {"available": False, "reason": BLACK_BOX_REASON}

    # ---- tilted / tail / worst-region risk ---------------------------------------------
    risk_to_tilt = {k: float(v) for k, v in cfg["tilt"]["risk_to_tilt"].items()}
    t_report = risk_to_tilt.get(risk_level, 0.0)
    if y_query is not None:
        Xe, ye, yhe, eval_label = X_query, y_query, yhat_query, "labelled query set"
    else:
        Xe, ye, yhe, eval_label = X_cal, y_cal, yhat_cal, "calibration set (queries unlabelled)"
    abs_e = np.abs(ye - yhe)
    tcfg = cfg["tilt"]
    tilt_info = {
        "t_report": t_report,
        "mapping": risk_to_tilt,
        "training_tilt": training_tilt,
        "eval_label": eval_label,
        "rmse": M.rmse(ye, yhe),
        "tilted_risk": M.tilted_risk(abs_e, t_report),
        "tail_fraction": float(tcfg["tail_fraction"]),
        "tail_risk": M.tail_risk(abs_e, float(tcfg["tail_fraction"])),
        "worst_region": M.worst_region_error(Xe, ye, yhe, int(tcfg["worst_region_bins"]), int(tcfg["min_bin_count"])),
        "feature_names": feature_names,
    }
    summary["tilt"] = tilt_info

    inputs = ReportInputs(
        context_of_use=context_of_use,
        risk_level=risk_level,
        model_name=model_name,
        mode=mode,
        checks=checks or [],
        validation=validation,
        alpha=engine.alpha,
        method=method,
        method_labels=METHOD_LABELS,
        method_table=mtab,
        coverage_source=coverage_source,
        envelope={
            "k": engine.ad.k,
            "percentile": engine.ad.percentile,
            "threshold": engine.ad.threshold,
            "n_train": int(len(X_train)),
            "features": feature_names,
            "pct_outside": summary["pct_outside"],
            "pct_far_outside": summary["pct_far_outside"],
            "flag_pr": summary["flag_pr"],
            "error_tolerance": err_tol,
            "flag_mode": engine.flag_mode,
            "pvalue_beta": engine.pvalue_beta,
            "n_cal": int(len(engine.ad.cal_distances)),
            "pct_flag_percentile": summary["pct_flag_percentile"],
            "pct_flag_pvalue": summary["pct_flag_pvalue"],
            "pct_flag_bh": summary["pct_flag_bh"],
        },
        gate={"tolerance": engine.gate_cfg.tolerance, "escalate_score": engine.gate_cfg.escalate_score, "counts": counts},
        fragility=fragility,
        tilt=tilt_info,
        notes=notes,
        coverage_stats=summary["coverage_stats"],
        shift=summary["shift"],
        group_table=group_table,
        group_info=group_info,
    )
    return PipelineResult(
        queries=df,
        engine=engine,
        method=method,
        method_table=mtab,
        coverage_source=coverage_source,
        summary=summary,
        report_inputs=inputs,
        report_md=build_report(inputs, cfg),
        fragility_available=white_box,
        notes=notes,
    )


# ---------------------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------------------


@dataclass
class DemoRun:
    data: DemoData
    model: object
    result: PipelineResult
    slice_df: pd.DataFrame


def demo_checks(data: DemoData, yhat_cal: np.ndarray, yhat_test: np.ndarray, cfg: dict[str, Any]) -> list[CheckResult]:
    """Verification checks on the demo data, framed the same way as upload mode."""
    f = data.feature_names
    frames = {
        "training": pd.DataFrame(np.column_stack([data.X_train, data.y_train]), columns=f + ["y"]),
        "calibration": pd.DataFrame(np.column_stack([data.X_cal, data.y_cal, yhat_cal]), columns=f + ["y_true", "y_pred"]),
        "query": pd.DataFrame(np.column_stack([data.X_test, data.y_test, yhat_test]), columns=f + ["y_true", "y_pred"]),
    }
    required = {"training": f + ["y"], "calibration": f + ["y_true", "y_pred"], "query": f + ["y_pred"]}
    return run_verification(frames, required, cfg.get("bounds"), cfg.get("units"))


def run_demo(
    cfg: dict[str, Any],
    model_kind: str = "mlp",
    use_fragility: bool = False,
    train_tilt: float = 0.0,
    risk_level: str = "medium",
    context_of_use: str = "",
    data: DemoData | None = None,
    model: object | None = None,
    shift: dict[str, Any] | None = None,
) -> DemoRun:
    """Train on [0,5]^3, calibrate, and evaluate on a test set with x1 up to 8."""
    seed = int(cfg.get("seed", 0))
    data = data or make_demo_data(seed)
    if model is None:
        model = make_surrogate(model_kind, cfg, tilt=train_tilt if model_kind == "mlp" else 0.0, seed=seed)
        model.fit(data.X_train, data.y_train)
    yhat_cal, yhat_test, yhat_ref = (model.predict(X) for X in (data.X_cal, data.X_test, data.X_ref))
    result = run_pipeline(
        feature_names=data.feature_names,
        X_train=data.X_train,
        X_cal=data.X_cal,
        y_cal=data.y_cal,
        yhat_cal=yhat_cal,
        X_query=data.X_test,
        yhat_query=yhat_test,
        y_query=data.y_test,
        model=model,
        cfg=cfg,
        risk_level=risk_level,
        context_of_use=context_of_use,
        use_fragility=use_fragility,
        checks=demo_checks(data, yhat_cal, yhat_test, cfg),
        reference=(data.y_ref, yhat_ref, "held-out reference set inside the training region [0,5]^3"),
        model_name=getattr(model, "name", type(model).__name__),
        mode="demo",
        training_tilt=getattr(model, "tilt", None) if is_white_box(model) else None,
        shift=shift,
    )
    Xs = slice_grid()
    sigma_s = None
    if result.engine.normalized is not None:
        fr = cfg["fragility"]
        sigma_s = perturbation_sigma(model, Xs, K=int(fr["K"]), tau=float(fr["tau"]), seed=seed)
    slice_df = result.engine.evaluate(Xs, model.predict(Xs), method=result.method, sigma=sigma_s, feature_names=data.feature_names)
    slice_df["f_true"] = simulator(Xs)
    return DemoRun(data=data, model=model, result=result, slice_df=slice_df)


def tilt_comparison(
    cfg: dict[str, Any],
    data: DemoData | None = None,
    tilts: list[float] | None = None,
    corrupt_fraction: float | None = None,
) -> pd.DataFrame:
    """Train the MLP with several tilts on clean and on partially corrupted training labels.

    After each training the split-conformal intervals are recalibrated on the (clean)
    calibration set. Accuracy metrics are computed on test points inside the training
    range against clean observations, so they measure how much the outliers distorted the fit.
    """
    seed = int(cfg.get("seed", 0))
    data = data or make_demo_data(seed)
    tilts = tilts if tilts is not None else [float(t) for t in cfg["tilt"]["demo_tilts"]]
    frac = float(corrupt_fraction if corrupt_fraction is not None else cfg["tilt"]["demo_corrupt_fraction"])
    y_bad, _ = corrupt_labels(data.y_train, frac, np.random.default_rng(seed + 1))
    in_range = data.X_test[:, 0] <= 5.0
    alpha = float(cfg["alpha"])
    tcfg = cfg["tilt"]
    rows = []
    for label_set, y_tr in (("clean", data.y_train), (f"{frac:.0%} corrupted", y_bad)):
        for t in tilts:
            model = make_surrogate("mlp", cfg, tilt=t, seed=seed).fit(data.X_train, y_tr)
            yhat_test = model.predict(data.X_test)
            sc = SplitConformal(alpha).fit(data.y_cal, model.predict(data.X_cal))
            iv = sc.predict(yhat_test)
            e_in = np.abs(data.y_test[in_range] - yhat_test[in_range])
            rows.append({
                "labels": label_set,
                "train_tilt": t,
                "rmse_in_range": M.rmse(data.y_test[in_range], yhat_test[in_range]),
                "tail_risk_in_range": M.tail_risk(e_in, float(tcfg["tail_fraction"])),
                "worst_region_rmse_in_range": M.worst_region_error(
                    data.X_test[in_range], data.y_test[in_range], yhat_test[in_range],
                    int(tcfg["worst_region_bins"]), int(tcfg["min_bin_count"]),
                )["rmse"],
                "coverage_in_range": M.coverage(data.y_test[in_range], iv.lower[in_range], iv.upper[in_range]),
                "coverage_overall": M.coverage(data.y_test, iv.lower, iv.upper),
                "interval_width": 2 * float(sc.q),
            })
    return pd.DataFrame(rows)
