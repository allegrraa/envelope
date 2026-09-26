"""Scripted 60-second demo: every number, caption, figure and narration line is computed here
from the same pipeline run (synthetic data, fixed seed). Nothing on screen is hard-coded.

The simulator design is fixed in advance (``envelope.data``): a stiffening term that only
activates for x1 > 5, while the surrogate is trained on [0, 5]. Parameters come from config.yaml
and are not tuned for the demo.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .data import TRAIN_RANGE, DemoData, make_demo_data  # noqa: E402
from .gate import Decision  # noqa: E402
from .pipeline import DemoRun, run_demo  # noqa: E402
from .report import assess_adequacy  # noqa: E402
from .surrogate import make_surrogate  # noqa: E402

CONTEXT = (
    "Screening surrogate for a structural-response simulator (illustrative). Predictions prioritise which "
    "design points get a full simulation run; they do not replace the simulator for final sign-off. "
    "Intended operating region: all inputs in [0, 5]."
)
SYNTHETIC_LABEL = "Synthetic demo data"
FINAL_GUARANTEE = "Guarantee holds when real queries resemble the calibration data. Far-out inputs are escalated, not fixed."
FINAL_DISCLAIMER = "Decision-support draft, not regulatory advice."
STAGES: dict[int, str] = {1: "Model looks perfect", 2: "The failure", 3: "Envelope on", 4: "Scoreboard", 5: "Report"}
TRAIN_MAX = TRAIN_RANGE[1]

INK, INK_2, GRID, BLUE = "#0b0b0b", "#3d3c3a", "#dcdad2", "#1f66c1"
DECISION_STYLE = {
    Decision.TRUST.value: ("TRUST", "#0ca30c", "o"),
    Decision.RERUN.value: ("RERUN", "#fab219", "s"),
    Decision.ESCALATE.value: ("ESCALATE", "#d03b3b", "^"),
}
FONT = {"base": 16, "tick": 15, "legend": 15, "title": 18}


@dataclass
class Story:
    numbers: dict[str, Any]
    slice_df: pd.DataFrame
    run: DemoRun


def _r(x: Any, nd: int = 6) -> Any:
    """Round floats for a stable, readable JSON."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else round(x, nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def _short_gap(g: str) -> str:
    return g.split(" -> ")[0]


def build_story(
    cfg: dict[str, Any],
    risk: str = "high",
    data: DemoData | None = None,
    model: object | None = None,
    shift: dict[str, Any] | None = None,
) -> Story:
    """Run the pipeline once and derive every number shown in Demo Mode."""
    seed = int(cfg.get("seed", 0))
    data = data or make_demo_data(seed)
    if model is None:
        model = make_surrogate("mlp", cfg, seed=seed).fit(data.X_train, data.y_train)
    run = run_demo(cfg, model_kind="mlp", risk_level=risk, context_of_use=CONTEXT, data=data, model=model, shift=shift)
    res, df = run.result, run.result.queries
    tol = float(cfg["error_tolerance"])
    in_rng = (df["x1"] <= TRAIN_MAX).to_numpy()
    err = df["abs_err"].to_numpy()
    y, yhat = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    rmse_in = float(np.sqrt(np.mean((y[in_rng] - yhat[in_rng]) ** 2)))
    rmse_out = float(np.sqrt(np.mean((y[~in_rng] - yhat[~in_rng]) ** 2)))
    covered = ((y >= df["lo_split"]) & (y <= df["hi_split"])).to_numpy()
    trusted = (df["decision"] == Decision.TRUST.value).to_numpy()
    wrong_out = (~in_rng) & (err > tol)
    n_out = int((~in_rng).sum())
    counts = {d.value: int((df["decision"] == d.value).sum()) for d in Decision}
    counts_out = {d.value: int(((df["decision"] == d.value).to_numpy() & ~in_rng).sum()) for d in Decision}
    row = res.method_table.set_index("method").loc["split"]
    verdict, gaps, _ = assess_adequacy(res.report_inputs, cfg)
    sl = run.slice_df
    end = sl.iloc[-1]
    sh = res.summary["shift"]
    law = res.summary["coverage_stats"]["law_nominal"]
    numbers = {
        "seed": seed,
        "risk_level": risk,
        "alpha": float(cfg["alpha"]),
        "target_coverage": 1 - float(cfg["alpha"]),
        "error_tolerance": tol,
        "train_range": list(TRAIN_RANGE),
        "n_train": len(data.X_train),
        "n_cal": len(data.X_cal),
        "n_test": len(df),
        "n_test_in_range": int(in_rng.sum()),
        "n_test_out_of_range": n_out,
        "rmse_in_range": rmse_in,
        "rmse_out_of_range": rmse_out,
        "rmse_ratio_out_vs_in": rmse_out / rmse_in,
        "max_error_out_of_range": float(err[~in_rng].max()),
        "interval_half_width": float(res.engine.split.q),
        "coverage_in_range": float(covered[in_rng].mean()),
        "coverage_out_of_range": float(covered[~in_rng].mean()),
        "slice_end": {"x1": float(end["x1"]), "true": float(end["f_true"]), "pred": float(end["y_pred"])},
        "gate_counts": counts,
        "gate_counts_out_of_range": counts_out,
        "scoreboard": {
            "trust_everything_pct": float(wrong_out.sum() / n_out),
            "trust_everything_n": int(wrong_out.sum()),
            "with_envelope_pct": float((wrong_out & trusted).sum() / n_out),
            "with_envelope_n": int((wrong_out & trusted).sum()),
            "n_out_of_range": n_out,
        },
        "coverage_table": {
            region: {
                "coverage": float(row[f"coverage_{region}"]),
                "ci_low": float(row[f"ci_low_{region}"]),
                "ci_high": float(row[f"ci_high_{region}"]),
                "n": int(row[f"n_{region}"]),
            }
            for region in ("inside", "outside", "overall")
        },
        "coverage_law": {"n_cal": int(law["n"]), "mean": float(law["mean"]), "p05": float(law["p05"])},
        "pct_outside_envelope": float(res.summary["pct_outside"]),
        "verdict": verdict,
        "gaps": [_short_gap(g) for g in gaps],
        "shift_check": {"status": sh["status"], "auc": float(sh["auc"]), "p_value": float(sh["p_value"])},
    }
    return Story(numbers=_r(numbers), slice_df=sl, run=run)


def numbers_json(story: Story) -> str:
    return json.dumps(story.numbers, indent=2, sort_keys=True)


# --------------------------------------------------------------------------------- captions
def caption(stage: int, n: dict[str, Any]) -> str:
    """One short sentence per stage, generated from the computed numbers."""
    sb = n["scoreboard"]
    c = n["gate_counts"]
    if stage == 1:
        return (f"Inside its training range the surrogate looks perfect: RMSE {n['rmse_in_range']:.2f} "
                f"on {n['n_test_in_range']} unseen test points.")
    if stage == 2:
        return (f"Confident, but wrong outside its training range: RMSE {n['rmse_out_of_range']:.1f} "
                f"there vs {n['rmse_in_range']:.2f} inside.")
    if stage == 3:
        return (f"The envelope sorts all {n['n_test']} queries: {c['TRUST']} trusted, "
                f"{c['RERUN_FULL_SIMULATION']} re-run, {c['ESCALATE_TO_HUMAN']} escalated to a human.")
    if stage == 4:
        return (f"Badly wrong out-of-range answers used anyway: {sb['trust_everything_pct']:.0%} if you trust everything, "
                f"{sb['with_envelope_pct']:.1%} with the envelope.")
    return (f"At {n['risk_level']} risk the credibility verdict is {n['verdict']}, "
            f"with {len(n['gaps'])} gap{'s' if len(n['gaps']) != 1 else ''} listed.")


# --------------------------------------------------------------------------------- figures
def _axes(figsize: tuple[float, float] = (14, 6.2)) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(labelsize=FONT["tick"], colors=INK_2)
    ax.grid(True, color="#eeede8", lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def stage_figure(stage: int, story: Story) -> Figure:
    """Main plot for stages 1-3 (identical axes across stages so the recording cuts cleanly)."""
    sl, n = story.slice_df, story.numbers
    x = sl["x1"].to_numpy()
    lo, hi = TRAIN_RANGE
    ymin = float(min(sl["f_true"].min(), sl["lower"].min())) - 1.0
    ymax = float(max(sl["f_true"].max(), sl["upper"].max())) + 1.0
    fig, ax = _axes()
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0)
    ax.text(lo + 0.08, ymax - 0.3, "training range [0, 5]", fontsize=FONT["base"], color=INK_2, va="top")
    show = x <= hi if stage == 1 else np.ones_like(x, dtype=bool)
    ax.plot(x[show], sl["f_true"].to_numpy()[show], color=INK, lw=3.2, label="true simulator")
    ax.plot(x[show], sl["y_pred"].to_numpy()[show], color=BLUE, lw=3.2, ls=(0, (5, 2)) if stage == 1 else "-",
            label="surrogate prediction")
    if stage == 2:
        e = n["slice_end"]
        ax.annotate("", xy=(e["x1"], e["pred"]), xytext=(e["x1"], e["true"]),
                    arrowprops=dict(arrowstyle="<->", color="#d03b3b", lw=2.5))
        ax.text(e["x1"] - 0.12, (e["true"] + e["pred"]) / 2, f"error {abs(e['true'] - e['pred']):.1f}",
                ha="right", va="center", fontsize=FONT["base"] + 2, color="#a12828", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    if stage >= 3:
        idx = np.linspace(0, len(sl) - 1, 27).round().astype(int)
        sub = sl.iloc[idx]
        yp = sub["y_pred"].to_numpy()
        ax.errorbar(sub["x1"], yp, yerr=[yp - sub["lower"].to_numpy(), sub["upper"].to_numpy() - yp],
                    fmt="none", ecolor=INK_2, elinewidth=1.6, capsize=4, zorder=3)
        for d, (label, color, marker) in DECISION_STYLE.items():
            m = (sub["decision"] == d).to_numpy()
            ax.scatter(sub["x1"].to_numpy()[m], yp[m], s=150, c=color, marker=marker, edgecolors=INK, linewidths=1.0,
                       zorder=4, label=f"{label} ({n['gate_counts'][d]} of {n['n_test']} test queries)")
    ax.set_xlim(0, 8.05)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x1   (x2 = x3 = 2.5)", fontsize=FONT["base"], color=INK_2)
    ax.set_ylabel("response", fontsize=FONT["base"], color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=FONT["legend"], frameon=False)
    fig.tight_layout()
    return fig


def _header(fig: Figure, stage: int, n: dict[str, Any]) -> None:
    fig.text(0.02, 0.975, f"Stage {stage}/5 · {STAGES[stage]}", fontsize=FONT["title"], fontweight="bold", color=INK, va="top")
    fig.text(0.98, 0.975, SYNTHETIC_LABEL, fontsize=FONT["base"], color="#7a5200", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fdf3dc", ec="none"))
    fig.text(0.02, 0.915, caption(stage, n), fontsize=FONT["base"] + 1, color=INK, va="top")


def stage_frame(stage: int, story: Story) -> Figure:
    """Static fallback frame for any stage (header + caption + content)."""
    n = story.numbers
    if stage <= 3:
        fig = stage_figure(stage, story)
        fig.set_size_inches(14, 7.6)
        fig.subplots_adjust(top=0.84)
        _header(fig, stage, n)
        return fig
    fig = plt.figure(figsize=(14, 7.6), facecolor="white")
    _header(fig, stage, n)
    if stage == 4:
        sb = n["scoreboard"]
        for i, (title, pct, cnt, color) in enumerate([
            ("Trust everything", sb["trust_everything_pct"], sb["trust_everything_n"], "#a12828"),
            ("With Envelope", sb["with_envelope_pct"], sb["with_envelope_n"], "#006300"),
        ]):
            x0 = 0.04 + i * 0.48
            fig.patches.append(plt.Rectangle((x0, 0.47), 0.44, 0.36, transform=fig.transFigure, fc="#f6f5f1", ec="none"))
            fig.text(x0 + 0.02, 0.79, title, fontsize=FONT["title"], fontweight="bold", color=INK, va="top")
            fig.text(x0 + 0.02, 0.71, f"{pct:.1%}", fontsize=44, fontweight="bold", color=color, va="top")
            fig.text(x0 + 0.02, 0.555, f"{cnt} of {sb['n_out_of_range']} out-of-range answers were off by more than\n"
                     f"{n['error_tolerance']:g} and trusted anyway", fontsize=FONT["base"] - 1, color=INK_2, va="top")
        ct = n["coverage_table"]
        rows = [["Interval coverage", "Target", "Actual", "95% CI", "Queries"]]
        for key, label in (("inside", "Inside envelope"), ("outside", "Outside envelope")):
            r = ct[key]
            rows.append([label, f"{n['target_coverage']:.0%}", f"{r['coverage']:.1%}",
                         f"{r['ci_low']:.1%} to {r['ci_high']:.1%}", str(r["n"])])
        ax = fig.add_axes((0.04, 0.05, 0.92, 0.34))
        ax.axis("off")
        tab = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.26, 0.12, 0.14, 0.3, 0.18])
        tab.auto_set_font_size(False)
        tab.set_fontsize(FONT["base"])
        tab.scale(1, 2.6)
        for (r_, _c), cell in tab.get_celld().items():
            cell.set_edgecolor("#dcdad2")
            if r_ == 0:
                cell.set_text_props(fontweight="bold")
    else:
        ok = n["verdict"] == "ADEQUATE"
        fig.patches.append(plt.Rectangle((0.04, 0.5), 0.92, 0.33, transform=fig.transFigure,
                                         fc="#e6f5e6" if ok else "#fae3e3", ec="none"))
        fig.text(0.06, 0.79, f"Verdict at {n['risk_level']} risk: {n['verdict']}", fontsize=30, fontweight="bold",
                 color="#006300" if ok else "#a12828", va="top")
        for i, g in enumerate(n["gaps"][:3]):
            fig.text(0.07, 0.69 - i * 0.055, f"• {g}", fontsize=FONT["base"] - 1, color=INK, va="top")
        sh = n["shift_check"]
        fig.text(0.04, 0.4, f"Guarantee validity check: {sh['status']} (AUC {sh['auc']:.2f}, p = {sh['p_value']:.3f})",
                 fontsize=FONT["base"], color=INK_2, va="top")
        fig.text(0.04, 0.29, FINAL_GUARANTEE.replace(". ", ".\n", 1), fontsize=FONT["base"] + 2, fontweight="bold",
                 color=INK, va="top", linespacing=1.5)
        fig.text(0.04, 0.12, FINAL_DISCLAIMER, fontsize=FONT["base"], color=INK_2, va="top")
    return fig


# --------------------------------------------------------------------------------- narration
def demo_script(n: dict[str, Any]) -> str:
    """Six-part, 60-second narration with the computed numbers filled in."""
    sb, c, ct = n["scoreboard"], n["gate_counts"], n["coverage_table"]
    gap = n["gaps"][0] if n["gaps"] else "no gaps were found"
    e = n["slice_end"]
    parts = [
        ("0-8s", "Stage 1 · Model looks perfect", "Open the app (Demo Mode). Nothing to click.",
         f"This surrogate stands in for an expensive simulator. Inside its training range, x1 from 0 to 5, it looks "
         f"perfect: RMSE {n['rmse_in_range']:.2f} on {n['n_test_in_range']} unseen test points."),
        ("8-20s", "Stage 2 · The failure", "Press Next (or the Right Arrow key).",
         f"Now ask about inputs up to 8. The real physics stiffens past 5 and the model doesn't know. It stays "
         f"confident and gets it wrong: RMSE {n['rmse_out_of_range']:.1f} out there versus {n['rmse_in_range']:.2f} inside, "
         f"and at x1 = {e['x1']:g} it is off by {abs(e['true'] - e['pred']):.1f}. Its conformal interval stays at "
         f"plus or minus {n['interval_half_width']:.2f} and covers only {n['coverage_out_of_range']:.0%} of out-of-range points."),
        ("20-35s", "Stage 3 · Envelope on", "Press Next.",
         f"Turn the envelope on. Every query gets a conformal interval and a distance to the training data. "
         f"Of {n['n_test']} test queries, {c['TRUST']} are trusted, {c['RERUN_FULL_SIMULATION']} go back to the simulator, "
         f"and {c['ESCALATE_TO_HUMAN']} are escalated to a human."),
        ("35-47s", "Stage 4 · Scoreboard", "Press Next.",
         f"If you trust everything, {sb['trust_everything_pct']:.0%} of out-of-range answers are off by more than "
         f"{n['error_tolerance']:g} and get used anyway. With the envelope that drops to {sb['with_envelope_pct']:.1%}. "
         f"Inside the envelope, coverage is {ct['inside']['coverage']:.1%}, 95% interval {ct['inside']['ci_low']:.1%} to "
         f"{ct['inside']['ci_high']:.1%}, against a {n['target_coverage']:.0%} target."),
        ("47-56s", "Stage 5 · Report", "Press Next, then click Generate report.",
         f"One click drafts the credibility report. At {n['risk_level']} risk the verdict is {n['verdict']}, and it says "
         f"why: {gap}."),
        ("56-60s", "Close", "Hold on the final screen.",
         f"{FINAL_GUARANTEE} {FINAL_DISCLAIMER}"),
    ]
    out = [
        "# envelope: 60-second demo script",
        "",
        f"All numbers were computed by `run_demo.py` from synthetic demo data (seed {n['seed']}, alpha {n['alpha']:g}, "
        f"risk level {n['risk_level']}). Re-run it after any change; do not edit numbers by hand.",
        "",
    ]
    for t, title, action, say in parts:
        out += [f"## {t} · {title}", "", f"**On screen:** {action}", "", f"**Say:** \"{say}\"", ""]
    return "\n".join(out)
