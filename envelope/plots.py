"""Matplotlib figures for the demo, the CLI and the Streamlit app."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .gate import Decision  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (fixed order)
# Status colours are paired with a distinct marker shape + label, never colour alone.
DECISION_STYLE = {
    Decision.TRUST.value: {"color": "#0ca30c", "marker": "o", "label": "TRUST"},
    Decision.RERUN.value: {"color": "#fab219", "marker": "s", "label": "RERUN_FULL_SIMULATION"},
    Decision.ESCALATE.value: {"color": "#d03b3b", "marker": "^", "label": "ESCALATE_TO_HUMAN"},
}
METHOD_COLORS = {"split": SERIES[0], "adaptive": SERIES[1], "fragility": SERIES[2]}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _fig(w: float = 9, h: float = 5, ncols: int = 1) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes.ravel()


def _decision_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, decisions: np.ndarray, size: float = 36) -> None:
    for d, st in DECISION_STYLE.items():
        m = decisions == d
        if m.any():
            ax.scatter(x[m], y[m], c=st["color"], marker=st["marker"], s=size, label=st["label"],
                       edgecolors=SURFACE, linewidths=0.8, zorder=4)


def plot_slice(slice_df: pd.DataFrame, feature: str = "x1", train_max: float = 5.0, title: str | None = None) -> Figure:
    """True function vs surrogate + intervals along a 1-D slice, points coloured by gate decision."""
    fig, (ax,) = _fig()
    x = slice_df[feature].to_numpy()
    ax.axvspan(x.min(), train_max, color="#e1e0d9", alpha=0.55, lw=0, label="training region")
    ax.fill_between(x, slice_df["lower"], slice_df["upper"], color=SERIES[0], alpha=0.18, lw=0, label="conformal interval")
    ax.plot(x, slice_df["f_true"], color=INK, lw=2, label="true simulator")
    ax.plot(x, slice_df["y_pred"], color=SERIES[0], lw=2, label="surrogate")
    step = max(1, len(x) // 40)
    sub = slice_df.iloc[::step]
    _decision_scatter(ax, sub[feature].to_numpy(), sub["y_pred"].to_numpy(), sub["decision"].to_numpy())
    ax.set_xlabel(f"{feature} (other inputs fixed at 2.5)")
    ax.set_ylabel("response y")
    ax.set_title(title or "Surrogate vs simulator: trained on [0, 5], queried to 8", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_error_vs_score(df: pd.DataFrame, escalate_score: float, error_tol: float) -> Figure:
    """|error| vs envelope score: the outside-envelope flag should catch the large errors."""
    fig, (ax,) = _fig()
    _decision_scatter(ax, df["env_score"].to_numpy(), np.maximum(df["abs_err"].to_numpy(), 1e-4), df["decision"].to_numpy(), 18)
    ax.set_yscale("log")
    ax.axvline(1.0, color=INK_2, lw=1, ls="--")
    ax.axvline(escalate_score, color=INK_2, lw=1, ls=":")
    ax.axhline(error_tol, color=MUTED, lw=1)
    trans = ax.get_xaxis_transform()
    ax.text(1.0, 0.99, "envelope edge ", ha="right", va="top", fontsize=8, color=INK_2, transform=trans)
    ax.text(escalate_score, 0.99, " escalate", ha="left", va="top", fontsize=8, color=INK_2, transform=trans)
    ax.text(ax.get_xlim()[1], error_tol, f"error tolerance {error_tol:g} ", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("envelope score (kNN distance / threshold)")
    ax.set_ylabel("|prediction error| (log)")
    ax.set_title("Large errors live outside the envelope", loc="left", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_coverage(method_table: pd.DataFrame, alpha: float, labels: dict[str, str] | None = None) -> Figure:
    """Grouped bars: empirical coverage per method, overall / inside / outside, vs nominal."""
    fig, (ax,) = _fig(9, 4.5)
    regions = ["overall", "inside", "outside"]
    n_m = len(method_table)
    width = 0.8 / max(n_m, 1)
    xs = np.arange(len(regions))
    for i, (_, r) in enumerate(method_table.iterrows()):
        vals = [r[f"coverage_{reg}"] for reg in regions]
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.92, color=METHOD_COLORS.get(r["method"], SERIES[0]),
               label=(labels or {}).get(r["method"], r["method"]))
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK_2)
    ax.axhline(1 - alpha, color=INK, lw=1.2, ls="--")
    ax.text(-0.5, 1 - alpha + 0.01, f"nominal {1 - alpha:.0%}", ha="left", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs, ["overall", "inside envelope", "outside envelope"])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage holds inside the envelope, collapses outside", loc="left", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=n_m)
    fig.tight_layout()
    return fig


def plot_tilt_comparison(df: pd.DataFrame) -> Figure:
    """In-range RMSE and tail risk by training tilt, clean vs corrupted labels."""
    fig, axes = _fig(10, 4.2, ncols=2)
    label_sets = list(dict.fromkeys(df["labels"]))
    tilts = list(dict.fromkeys(df["train_tilt"]))
    xs = np.arange(len(tilts))
    w = 0.8 / len(label_sets)
    for ax, col, title in zip(axes, ["rmse_in_range", "tail_risk_in_range"], ["RMSE (in range)", "Tail risk: mean worst 10% |error|"]):
        for i, ls in enumerate(label_sets):
            sub = df[df["labels"] == ls].set_index("train_tilt").loc[tilts]
            pos = xs - 0.4 + w * (i + 0.5)
            ax.bar(pos, sub[col], w * 0.92, color=SERIES[i], label=f"{ls} labels")
            for p, v in zip(pos, sub[col]):
                ax.text(p, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(xs, [f"t = {t:g}" for t in tilts])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("training tilt")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Tilted ERM: negative tilt resists corrupted labels", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def plot_query_intervals(df: pd.DataFrame, feature: str) -> Figure:
    """Upload mode: predictions with intervals against one feature, coloured by decision."""
    fig, (ax,) = _fig()
    d = df.sort_values(feature)
    ax.vlines(d[feature], d["lower"], d["upper"], color=SERIES[0], alpha=0.35, lw=1, label="conformal interval")
    if "y_true" in d:
        ax.scatter(d[feature], d["y_true"], s=10, c=INK, marker="x", lw=0.8, label="y_true", zorder=3)
    _decision_scatter(ax, d[feature].to_numpy(), d["y_pred"].to_numpy(), d["decision"].to_numpy(), 22)
    ax.set_xlabel(feature)
    ax.set_ylabel("prediction")
    ax.set_title("Query predictions, intervals and gate decisions", loc="left", fontsize=12)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_gate_view(slice_df: pd.DataFrame, feature: str = "x1", train_max: float = 5.0, n_points: int = 33) -> Figure:
    """Single-screen UI plot: true function, training region, predictions with interval error bars,
    points coloured green / yellow / red by gate decision."""
    fig, (ax,) = _fig(8.5, 5.2)
    x = slice_df[feature].to_numpy()
    ax.axvspan(x.min(), train_max, color=GRID, alpha=0.6, lw=0, label="training region")
    ax.plot(x, slice_df["f_true"], color=INK, lw=2, label="true function")
    ax.plot(x, slice_df["y_pred"], color=SERIES[0], lw=1.5, alpha=0.8, label="model prediction")
    idx = np.linspace(0, len(slice_df) - 1, n_points).round().astype(int)
    sub = slice_df.iloc[idx]
    yp = sub["y_pred"].to_numpy()
    ax.errorbar(sub[feature], yp, yerr=[yp - sub["lower"].to_numpy(), sub["upper"].to_numpy() - yp],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, alpha=0.7, label="prediction interval", zorder=3)
    _decision_scatter(ax, sub[feature].to_numpy(), yp, sub["decision"].to_numpy(), size=46)
    # keep all three decisions in the legend even when one is absent at these settings
    present = set(sub["decision"])
    for d, st_ in DECISION_STYLE.items():
        if d not in present:
            ax.scatter([], [], c=st_["color"], marker=st_["marker"], s=46, label=st_["label"])
    ax.set_xlabel(f"{feature}  (other inputs fixed at 2.5)")
    ax.set_ylabel("response")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def plot_coverage_by_bin(table: pd.DataFrame, alpha: float) -> Figure:
    """Coverage per envelope-score bin: group-conditional vs global split conformal, with CP intervals."""
    fig, (ax,) = _fig(9, 4.6)
    xs = np.arange(len(table))
    w = 0.38
    ax.bar(xs - w / 2, table["coverage_split"], w * 0.92, color=SERIES[0], label="global split conformal")
    ax.bar(xs + w / 2, table["coverage_group"], w * 0.92, color=SERIES[2], label="group-conditional")
    err = np.vstack([table["coverage_group"] - table["ci_low"], table["ci_high"] - table["coverage_group"]])
    ax.errorbar(xs + w / 2, table["coverage_group"], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for x, (_, r) in zip(xs, table.iterrows()):
        if not np.isnan(r["coverage_group"]):
            ax.text(x + w / 2, min(r["ci_high"], 1.0) + 0.02, f"{r['coverage_group']:.0%}", ha="center", fontsize=8, color=INK_2)
    ax.axhline(1 - alpha, color=INK, lw=1.2, ls="--")
    ax.text(-0.5, 1 - alpha + 0.01, f"nominal {1 - alpha:.0%}", ha="left", va="bottom", fontsize=8, color=INK)
    labels = [
        f"bin {int(r['bin'])}\nscore {r['score_low']:.2f}-{r['score_high']:.2f}\n"
        f"{int(r['n_cal'])} cal pts" + ("\n(global fallback)" if r["quantile_source"] != "bin" else "")
        for _, r in table.iterrows()
    ]
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage by envelope-score bin", loc="left", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig
