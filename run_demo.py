"""End-to-end demo: train on [0,5], query to 8, show the failure, the flags, the report verdict.

Writes plots, tables, reports, example upload CSVs and the API state file (state.npz) to ./out/.
Usage: python run_demo.py [--risk high] [--fast]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envelope.config import load_config
from envelope.data import make_demo_data
from envelope.pipeline import METHOD_LABELS, run_demo, tilt_comparison
from envelope.plots import plot_coverage, plot_coverage_by_bin, plot_error_vs_score, plot_slice, plot_tilt_comparison
from envelope.report import assess_adequacy
from envelope.state import save_state
from envelope.demo_story import build_story, demo_script, numbers_json, stage_frame
from envelope.surrogate import make_surrogate, save_mlp

CONTEXT = (
    "Screening surrogate for a structural-response simulator (illustrative). Predictions are used to "
    "prioritise which design points get a full simulation run; they do not replace the simulator for "
    "final sign-off. Intended operating region: all inputs in [0, 5]."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--risk", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--fast", action="store_true", help="fewer MLP epochs (quicker, slightly worse fits)")
    args = ap.parse_args()

    overrides = {"surrogate": {"mlp_epochs": 400}} if args.fast else None
    cfg = load_config(args.config, overrides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = make_demo_data(int(cfg["seed"]))

    print("== 1. Train an MLP surrogate on [0,5]^3; query x1 up to 8 ==")
    mlp = make_surrogate("mlp", cfg, seed=int(cfg["seed"])).fit(data.X_train, data.y_train)
    save_mlp(mlp, str(out / "surrogate_mlp.pt"), data.X_train, data.y_train)  # reused by the app (no training delay)
    run = run_demo(cfg, model_kind="mlp", use_fragility=False, risk_level=args.risk, context_of_use=CONTEXT, data=data, model=mlp)
    res = run.result
    df = res.queries
    pd.set_option("display.width", 160, "display.precision", 3)
    print(f"model: {run.model.name}")
    print(res.method_table.assign(method=res.method_table["method"].map(METHOD_LABELS)).to_string(index=False))

    print("\n== 2. The failure: naive split-conformal coverage outside the training range ==")
    row = res.method_table.set_index("method").loc["split"]
    print(f"nominal {1 - cfg['alpha']:.0%} | inside envelope {row['coverage_inside']:.1%} | outside envelope {row['coverage_outside']:.1%}")

    print("\n== 3. The flags ==")
    s = res.summary
    pr = s["flag_pr"]
    print(f"{s['pct_outside']:.1f}% of queries outside envelope; gate: {s['gate_counts']}")
    print(f"outside-envelope flag vs |error| > {cfg['error_tolerance']}: precision {pr['precision']:.1%}, recall {pr['recall']:.1%}")
    frag = res.report_inputs.fragility
    print(f"fragility sigma: inside {frag['mean_sigma_inside']:.4f} vs outside {frag['mean_sigma_outside']:.4f}")

    print("\n== 4. Report verdict ==")
    verdict, gaps, _ = assess_adequacy(res.report_inputs, cfg)
    print(f"risk level {args.risk.upper()}: {verdict}")
    for g in gaps:
        print(f"  - {g}")

    # Fragility-normalised intervals driving the gate
    run_fr = run_demo(cfg, use_fragility=True, risk_level=args.risk, context_of_use=CONTEXT, data=data, model=run.model)
    # Black-box baseline: fragility disabled
    run_gbr = run_demo(cfg, model_kind="gbr", risk_level=args.risk, context_of_use=CONTEXT, data=data)

    print("\n== 5. Tilted ERM comparison (clean vs 5% corrupted training labels) ==")
    tilt_df = tilt_comparison(cfg, data)
    print(tilt_df.to_string(index=False))

    # ---- write artefacts ----
    plot_slice(run.slice_df).savefig(out / "slice.png", dpi=150)
    plot_slice(run_fr.slice_df, title="Fragility-normalised intervals along the slice").savefig(out / "slice_fragility.png", dpi=150)
    plot_error_vs_score(df, float(cfg["gate"]["escalate_score"]), float(cfg["error_tolerance"])).savefig(out / "error_vs_score.png", dpi=150)
    plot_coverage(res.method_table, float(cfg["alpha"]), METHOD_LABELS).savefig(out / "coverage.png", dpi=150)
    plot_tilt_comparison(tilt_df).savefig(out / "tilt_comparison.png", dpi=150)
    plot_coverage_by_bin(res.summary["group_table"], float(cfg["alpha"])).savefig(out / "coverage_by_bin.png", dpi=150)
    res.summary["group_table"].to_csv(out / "coverage_by_bin.csv", index=False)
    (out / "report.md").write_text(res.report_md)
    (out / "report_fragility.md").write_text(run_fr.result.report_md)
    (out / "report_blackbox_gbr.md").write_text(run_gbr.result.report_md)
    save_state(
        out / "state.npz",
        cfg=cfg,
        feature_names=data.feature_names,
        method=res.method,
        X_train=data.X_train,
        X_cal=data.X_cal,
        y_cal=data.y_cal,
        yhat_cal=run.model.predict(data.X_cal),
        threshold=res.engine.ad.threshold,
        report_path=out / "report.md",
    )
    df.to_csv(out / "queries.csv", index=False)
    res.method_table.to_csv(out / "method_table.csv", index=False)
    tilt_df.to_csv(out / "tilt_comparison.csv", index=False)

    # Example CSVs for upload mode (the GBR's predictions, treated as an external black-box model)
    up = out / "upload_example"
    up.mkdir(exist_ok=True)
    f = data.feature_names
    gbr = run_gbr.model
    pd.DataFrame(data.X_train, columns=f).to_csv(up / "training.csv", index=False)
    cal = pd.DataFrame(data.X_cal, columns=f).assign(y_true=data.y_cal, y_pred=gbr.predict(data.X_cal))
    cal.to_csv(up / "calibration.csv", index=False)
    q = pd.DataFrame(data.X_test, columns=f).assign(y_pred=gbr.predict(data.X_test), y_true=data.y_test)
    q.to_csv(up / "query.csv", index=False)
    q.drop(columns="y_true").to_csv(up / "query_unlabelled.csv", index=False)

    # Demo Mode artefacts: on-screen numbers, narration with real numbers, static fallback frames
    story = build_story(cfg, risk="high", data=data, model=mlp, shift=res.summary["shift"])
    (out / "demo_numbers.json").write_text(numbers_json(story) + "\n")
    (out / "demo_script.md").write_text(demo_script(story.numbers))
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for k in range(1, 6):
        fig = stage_frame(k, story)
        fig.savefig(frames / f"stage{k}.png", dpi=120)
        plt.close(fig)
    print("\n== 6. Demo Mode numbers ==")
    print(f"scoreboard: {story.numbers['scoreboard']}")

    print(f"\nWrote plots, reports and tables to {out.resolve()} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
