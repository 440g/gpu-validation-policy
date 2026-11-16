"""Report figures, drawn from the CSVs in results/."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .evaluation import EXPLORATORY

COLORS = {"glm": "#2a78d6", "hgb": "#eb6834", "random": "#8a8984", "oracle": "#52514e"}
NAMES = {"glm": "GLM", "hgb": "Gradient boosting", "random": "Random (same cap)", "oracle": "Oracle ceiling"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def headline(out):
    t = pd.read_csv(out / "test_table.csv").set_index("quantity")
    curve = pd.read_csv(out / "test_d_curve.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})

    order = ["oracle", "glm", "hgb", "random"]
    for i, k in enumerate(order):
        est, lo, hi = t.loc[k, ["estimate", "lo", "hi"]]
        a.plot([lo, hi], [i, i], color=COLORS[k], lw=2, solid_capstyle="round")
        a.plot(est, i, "o", color=COLORS[k], ms=6, mec="white", mew=1.5)
        a.text(hi + 80, i, f"{est:,.0f}", va="center", color=INK)
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_yticks(range(len(order)), [NAMES[k] for k in order])
    a.invert_yaxis()
    a.set_xlabel("net saving, min per 1,000 test jobs (95% day-block CI)")
    a.grid(axis="y", visible=False)

    for k in ("glm", "hgb", "random"):
        b.plot(curve["d"], curve[k], color=COLORS[k], lw=2, marker="o", ms=4, mec="white", label=NAMES[k])
    b.axhline(0, color=MUTED, lw=0.8)
    b.axvspan(min(C.D_GRID), max(C.D_GRID), color=GRID, alpha=0.6, lw=0)
    b.text(np.sqrt(min(C.D_GRID) * max(C.D_GRID)), b.get_ylim()[0] * 0.97, "locked d grid", color=MUTED,
           fontsize=7, ha="center", va="bottom")
    b.set_xscale("log")
    b.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20, 30], ["0.25", "0.5", "1", "2", "5", "10", "20", "30"])
    b.set_xlabel(f"validation cost d (min), r = {C.R_BASE}")
    b.set_ylabel("min per 1,000 jobs")
    b.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.32, 0.02))
    fig.tight_layout()
    fig.savefig(out / "fig_headline.png", dpi=200)
    plt.close(fig)


def slices(out):
    s = pd.read_csv(out / "test_slices.csv")
    s = s[~s["variable"].str.contains(EXPLORATORY)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(s))
    for off, m in ((-0.17, "glm"), (0.17, "hgb")):
        lo = s[f"{m}_share_lo"].clip(lower=-0.1)
        ax.hlines(y + off, lo, s[f"{m}_share_hi"], color=COLORS[m], lw=2)
        ax.plot(s[f"{m}_oracle_share"].clip(lower=-0.1), y + off, "o", color=COLORS[m], ms=5, mec="white",
                mew=1.2, label=NAMES[m])
    labels = [f"{v.replace('_', ' ')}: {l}   (n={n:,}, fail={f:,}){'  *' if f < C.MIN_SLICE_FAILED else ''}"
              for v, l, n, f in zip(s["variable"], s["level"], s["n"], s["n_failed"])]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.02)
    ax.set_xlabel("share of the oracle's achievable saving captured (95% day-block CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_slices.png", dpi=200)
    plt.close(fig)


def draw(out=C.RESULTS):
    headline(out)
    slices(out)
