"""Markdown credibility-report generator with illustrative adequacy rules."""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data import CheckResult

DISCLAIMER = (
    "**Disclaimer.** This is a decision-support *draft* generated automatically. All thresholds "
    "and adequacy rules are illustrative defaults chosen for demonstration. This report is not "
    "regulatory advice, not a certification, and not a claim of compliance with any standard. "
    "Conformal coverage guarantees assume exchangeable calibration and query data and weaken or "
    "fail under real distribution shift. A qualified reviewer must judge adequacy for the actual "
    "context of use."
)


@dataclass
class ReportInputs:
    """Everything needed to render the credibility report."""

    context_of_use: str
    risk_level: str
    model_name: str
    mode: str
    checks: list[CheckResult]
    validation: dict[str, Any]
    alpha: float
    method: str
    method_labels: dict[str, str]
    method_table: pd.DataFrame
    coverage_source: str
    envelope: dict[str, Any]
    gate: dict[str, Any]
    fragility: dict[str, Any]
    tilt: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    coverage_stats: dict[str, Any] | None = None
    shift: dict[str, Any] | None = None
    group_table: pd.DataFrame | None = None
    group_info: dict[str, Any] | None = None


def _f(x: Any, fmt: str = ".3f") -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "n/a"
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def _pct(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def selected_coverage(inputs: ReportInputs) -> float:
    """Empirical overall coverage of the method that drives the gate."""
    row = inputs.method_table[inputs.method_table["method"] == inputs.method]
    return float(row["coverage_overall"].iloc[0]) if len(row) else float("nan")


def assess_adequacy(inputs: ReportInputs, cfg: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Apply the illustrative rule set for the chosen risk level.

    Returns (verdict, gaps, passed_rules). Verdict is ``ADEQUATE`` or ``INSUFFICIENT``.
    """
    rules = cfg["adequacy"][inputs.risk_level]
    nominal = 1 - inputs.alpha
    gaps: list[str] = []
    passed: list[str] = []

    failed_checks = [c for c in inputs.checks if c.status == "FAIL"]
    if failed_checks:
        gaps.append(
            "Verification: " + ", ".join(f"{c.name} ({c.detail})" for c in failed_checks)
            + " -> fix the data pipeline and re-run."
        )
    elif not inputs.checks:
        gaps.append("Verification: no data checks were run -> run schema/NaN/unit checks.")
    else:
        passed.append("Verification checks have no failures.")

    cov = selected_coverage(inputs)
    gap_max = float(rules["max_coverage_gap"])
    if math.isnan(cov):
        gaps.append("Uncertainty quantification: empirical coverage could not be measured -> provide labelled data.")
    elif abs(cov - nominal) > gap_max:
        gaps.append(
            f"Uncertainty quantification: empirical coverage {cov:.1%} deviates from nominal {nominal:.0%} by "
            f"{abs(cov - nominal) * 100:.1f} points (allowed {gap_max * 100:.0f}) -> restrict queries to the "
            "envelope, recalibrate on data representative of the queries, or collect more calibration data."
        )
    else:
        passed.append(f"Empirical coverage {cov:.1%} within {gap_max * 100:.0f} points of nominal {nominal:.0%}.")
    if "calibration set" in inputs.coverage_source:
        gaps.append(
            "Uncertainty quantification: coverage was estimated on calibration splits only (queries unlabelled), "
            "so it does not reflect out-of-envelope behaviour -> obtain reference results for a sample of queries."
        )

    pct_out = float(inputs.envelope["pct_outside"])
    max_out = float(rules["max_pct_outside"])
    if pct_out > max_out:
        gaps.append(
            f"Applicability: {pct_out:.1f}% of queries are outside the envelope (allowed {max_out:.0f}%) -> "
            "run the full simulation for flagged queries, or extend training data to cover them."
        )
    else:
        passed.append(f"{pct_out:.1f}% of queries outside the envelope (allowed {max_out:.0f}%).")

    r2 = float(inputs.validation["r2"])
    min_r2 = float(rules["min_r2"])
    if math.isnan(r2) or r2 < min_r2:
        gaps.append(f"Validation: R2 = {_f(r2)} on reference data (required >= {min_r2}) -> improve the model or reference data.")
    else:
        passed.append(f"Reference R2 = {r2:.3f} (required >= {min_r2}).")

    verdict = "INSUFFICIENT" if gaps else "ADEQUATE"
    return verdict, gaps, passed


def _cov_ci(r: Any, region: str) -> str:
    cov = _pct(r[f"coverage_{region}"])
    lo, hi = r.get(f"ci_low_{region}", float("nan")), r.get(f"ci_high_{region}", float("nan"))
    if cov == "n/a" or lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return cov
    return f"{cov} [{100 * lo:.1f}, {100 * hi:.1f}]"


def _method_table_md(inputs: ReportInputs) -> str:
    lines = [
        "| Method | Coverage overall [95% CI] | Inside envelope [95% CI] | Outside envelope [95% CI] | Mean width overall | Width inside | Width outside |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in inputs.method_table.iterrows():
        label = inputs.method_labels.get(r["method"], r["method"])
        if r["method"] == inputs.method:
            label += " (drives gate)"
        lines.append(
            f"| {label} | {_cov_ci(r, 'overall')} | {_cov_ci(r, 'inside')} | {_cov_ci(r, 'outside')} "
            f"| {_f(r['width_overall'])} | {_f(r['width_inside'])} | {_f(r['width_outside'])} |"
        )
    return "\n".join(lines)


def build_report(inputs: ReportInputs, cfg: dict[str, Any]) -> str:
    """Render the credibility report as Markdown."""
    verdict, gaps, passed = assess_adequacy(inputs, cfg)
    rules = cfg["adequacy"][inputs.risk_level]
    nominal = 1 - inputs.alpha
    v = inputs.validation
    env = inputs.envelope
    fr = inputs.fragility
    tl = inputs.tilt
    out: list[str] = []
    add = out.append

    add("# Credibility Evidence Report (draft)")
    add("")
    add(f"*Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by envelope - mode: {inputs.mode} - model: {inputs.model_name}*")
    add("")
    add("> " + DISCLAIMER)
    add("")

    add("## 1. Context of Use")
    add("")
    add(inputs.context_of_use.strip() or "_No context of use was entered. The adequacy verdict is meaningless without one._")
    add("")

    add("## 2. Model Risk Level")
    add("")
    add(f"Selected risk level: **{inputs.risk_level.upper()}** (user-selected). Illustrative rules applied at this level:")
    add("")
    add(f"- empirical coverage within {rules['max_coverage_gap'] * 100:.0f} points of nominal")
    add(f"- at most {rules['max_pct_outside']:.0f}% of queries outside the envelope")
    add(f"- reference R2 >= {rules['min_r2']}")
    add("- no failed verification checks")
    add("")

    add("## 3. Verification (data checks)")
    add("")
    if inputs.checks:
        add("| Check | Status | Detail |")
        add("|---|---|---|")
        for c in inputs.checks:
            add(f"| {c.name} | {c.status} | {c.detail} |")
    else:
        add("_No verification checks were run._")
    add("")

    add("## 4. Validation")
    add("")
    add(f"Reference data: {v['reference_label']} (n = {v['n']}).")
    add("")
    add("| Data | n | RMSE | R2 |")
    add("|---|---|---|---|")
    add(f"| Reference | {v['n']} | {_f(v['rmse'])} | {_f(v['r2'])} |")
    for key, label in (("query_inside", "Queries inside envelope"), ("query_outside", "Queries outside envelope")):
        if key in v:
            add(f"| {label} | {v[key]['n']} | {_f(v[key]['rmse'])} | {_f(v[key]['r2'])} |")
    add("")

    add("### 4.1 Tail and tilted risk")
    add("")
    add(
        f"Evaluated on: {tl['eval_label']}. Tilted risk uses absolute errors in target units, "
        f"L_t = (1/t) log mean exp(t |e|), with t = **{tl['t_report']:g}** taken from the risk level "
        f"(mapping {', '.join(f'{k}: {v:g}' for k, v in tl['mapping'].items())})."
    )
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| RMSE | {_f(tl['rmse'])} |")
    add(f"| Tilted risk (t = {tl['t_report']:g}) | {_f(tl['tilted_risk'])} |")
    add(f"| Tail risk (mean of worst {tl['tail_fraction']:.0%} abs. errors) | {_f(tl['tail_risk'])} |")
    wr = tl["worst_region"]
    if wr.get("bounds"):
        region = ", ".join(f"{n} in [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in zip(tl["feature_names"], wr["bounds"]))
        add(f"| Worst-region RMSE | {_f(wr['rmse'])} ({region}; n = {wr['n']}) |")
    else:
        add("| Worst-region RMSE | n/a (too few points per bin) |")
    if tl.get("training_tilt") is not None:
        add(f"| Training objective tilt | t = {tl['training_tilt']:g} |")
    add("")
    add(
        "_The risk-level -> tilt mapping is an illustrative design choice, not an established standard. "
        "Positive t emphasises the worst errors (and can amplify outliers); negative t de-emphasises them._"
    )
    add("")

    add("## 5. Uncertainty Quantification")
    add("")
    add(
        f"Method driving the gate: **{inputs.method_labels.get(inputs.method, inputs.method)}**, "
        f"alpha = {inputs.alpha:g} (nominal coverage {nominal:.0%}), finite-sample quantile at level "
        "ceil((n+1)(1-alpha))/n of calibration nonconformity scores."
    )
    add("")
    add(f"Coverage measured on: {inputs.coverage_source}.")
    add("")
    sh = inputs.shift
    if sh:
        add(f"**Guarantee validity: {sh['status']}.** {sh['message']}")
        if sh["status"] != "SKIP":
            add(
                f"(Classifier two-sample test: logistic regression vs random forest chosen by 5-fold CV AUC; "
                f"held-out AUC and a {sh['n_perm']}-permutation p-value; OK requires AUC < {sh['auc_cutoff']:g} and "
                f"p > {sh['p_cutoff']:g}. A WARN means the coverage guarantee may not apply. This is a diagnostic, not a proof.)"
            )
        add("")
    add(_method_table_md(inputs))
    add("")
    cs = inputs.coverage_stats
    if cs:
        ln, lu = cs["law_nominal"], cs["law_used"]
        add(
            f"Intervals in brackets are {cs['ci_level']:.0%} Clopper-Pearson (exact binomial) intervals for the measured coverage. "
            f"Finite-sample expectation for split conformal with n = {cs['n_cal']} calibration points at alpha = {inputs.alpha:g}: "
            f"coverage over calibration draws ~ Beta({ln['n'] + 1 - ln['l']}, {ln['l']}), mean {ln['mean']:.1%}, "
            f"5th percentile {ln['p05']:.1%}, P(coverage >= {1 - inputs.alpha:.0%}) = {ln['prob_at_least_target']:.2f}."
        )
        add("")
        applied = "**applied**" if cs["correction_applied"] else "not applied (set coverage.small_sample_correction: true)"
        add(
            f"Small-sample correction (delta = {cs['delta']:g}): alpha' = {cs['alpha_corrected']:.4f} makes "
            f"P(coverage >= {1 - inputs.alpha:.0%}) >= {1 - cs['delta']:.0%}; {applied}."
            + (f" With alpha' the law has mean {lu['mean']:.1%} and 5th percentile {lu['p05']:.1%}." if cs["correction_applied"] else "")
        )
        add("")
    add(
        "_Split conformal guarantees marginal coverage only for queries exchangeable with the calibration data. "
        "Loss of coverage outside the envelope is expected and is exactly what the envelope flag is for._"
    )
    add("")

    gt, gi = inputs.group_table, inputs.group_info
    if gt is not None and gi is not None:
        add("### 5.1 Coverage by envelope-score bin (group-conditional)")
        add("")
        add(
            f"Queries are split into {gi['n_bins']} quantile bins of the envelope score. Each bin gets its own conformal "
            f"quantile from the calibration points in it; bins with fewer than {gi['min_cal_per_bin']} calibration points "
            "fall back to the global quantile and carry no bin-level guarantee."
        )
        add("")
        add("| Bin | Envelope score | Queries | Calibration pts | Quantile | Half-width | Coverage, group-conditional [95% CI] | Coverage, global split |")
        add("|---|---|---|---|---|---|---|---|")
        for _, r in gt.iterrows():
            ci = "" if math.isnan(r["ci_low"]) else f" [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
            add(
                f"| {int(r['bin'])} | {r['score_low']:.2f} to {r['score_high']:.2f} | {int(r['n_query'])} | {int(r['n_cal'])} | "
                f"{r['quantile_source']} | {r['half_width']:.3f} | {_pct(r['coverage_group'])}{ci} | {_pct(r['coverage_split'])} |"
            )
        add("")
        w = gi.get("worst")
        if w:
            add(
                f"Worst bin: **bin {int(w['bin'])}** (envelope score {w['score_low']:.2f} to {w['score_high']:.2f}), "
                f"coverage {w['coverage_group']:.1%} vs nominal {nominal:.0%}"
                + (" using the global fallback quantile (too few calibration points there)." if w["quantile_source"] != "bin" else ".")
            )
            add("")
    add("### 5.2 Sensitivity to parameter perturbation")
    add("")
    if fr.get("available"):
        ratio = fr["mean_sigma_outside"] / fr["mean_sigma_inside"] if fr["mean_sigma_inside"] else float("nan")
        add(
            f"K = {fr['K']} weight-perturbed copies; per-layer Gaussian noise with norm about tau = {fr['tau']:g} "
            f"times that layer's weight norm. sigma(x) = std of the perturbed predictions."
        )
        add("")
        add("| Region | Mean sigma(x) |")
        add("|---|---|")
        add(f"| Inside envelope | {_f(fr['mean_sigma_inside'], '.4f')} |")
        add(f"| Outside envelope | {_f(fr['mean_sigma_outside'], '.4f')} |")
        add(f"| Ratio outside / inside | {_f(ratio, '.2f')} |")
        add("")
        add(
            f"Fragility-normalised intervals (beta = {_f(fr['beta'], '.4g')}) are "
            + ("**used** to drive the gate." if fr["used_for_intervals"] else "reported for comparison only.")
        )
        add("")
        add(
            "_This is an empirical diagnostic of how sensitive predictions are to small weight changes. "
            "It is not a certified robustness bound, and a low sigma(x) does not imply a low error._"
        )
    else:
        add(f"**Not available.** {fr.get('reason', 'Model is black-box.')}")
    add("")

    add("## 6. Applicability (validity envelope)")
    add("")
    add(
        f"Envelope definition: features standardised on the training data (n = {env['n_train']}, features "
        f"{', '.join(env['features'])}); score = mean distance to the k = {env['k']} nearest training points "
        f"divided by the {env['percentile']:g}th percentile of leave-one-out training distances "
        f"(threshold = {env['threshold']:.3f}). Score <= 1: inside."
    )
    add("")
    if "flag_mode" in env:
        add(
            f"Conformal p-values: p(x) = (1 + #{{calibration kNN distances >= d(x)}}) / (n + 1) with n = {env['n_cal']} "
            f"calibration points; flag OUTSIDE if p < beta = {env['pvalue_beta']:g} (for in-distribution queries "
            f"P(p <= beta) <= beta). Benjamini-Hochberg controls the false-discovery rate of OUTSIDE flags over a batch. "
            f"Flag used by the gate: **{env['flag_mode']}**."
        )
        add("")
        add("| Outside flag | % of queries flagged |")
        add("|---|---|")
        add(f"| Percentile threshold (score > 1) | {env['pct_flag_percentile']:.1f}% |")
        add(f"| Conformal p-value < {env['pvalue_beta']:g} | {env['pct_flag_pvalue']:.1f}% |")
        add(f"| Benjamini-Hochberg at {env['pvalue_beta']:g} (batch) | {env['pct_flag_bh']:.1f}% |")
        add("")
    add(f"- Queries outside the envelope: **{env['pct_outside']:.1f}%**")
    add(f"- Queries far outside (score > {inputs.gate['escalate_score']:g}): {env['pct_far_outside']:.1f}%")
    pr = env.get("flag_pr")
    if pr:
        add(
            f"- Outside-envelope flag vs large errors (|error| > {env['error_tolerance']:g}): precision "
            f"{_pct(pr['precision'])}, recall {_pct(pr['recall'])} ({pr['n_large_errors']} large errors, "
            f"{pr['n_flagged']} flagged)"
        )
    add("")
    add("Gate decisions (TRUST requires inside the envelope and interval half-width <= "
        f"{inputs.gate['tolerance']:g}; slightly outside -> rerun; score > {inputs.gate['escalate_score']:g} -> escalate):")
    add("")
    add("| Decision | Count |")
    add("|---|---|")
    for k, c in inputs.gate["counts"].items():
        add(f"| {k} | {c} |")
    add("")

    add("## 7. Adequacy Verdict")
    add("")
    add(f"**{verdict}** for the stated context of use at risk level {inputs.risk_level.upper()} (illustrative rules).")
    add("")
    if passed:
        add("Rules met:")
        add("")
        for p in passed:
            add(f"- {p}")
        add("")
    if gaps:
        add("Gaps to close:")
        add("")
        for g in gaps:
            add(f"- {g}")
        add("")
    if inputs.notes:
        add("## Notes")
        add("")
        for n in dict.fromkeys(inputs.notes):
            add(f"- {n}")
        add("")
    add("---")
    add("")
    add(DISCLAIMER)
    add("")
    return "\n".join(out)
