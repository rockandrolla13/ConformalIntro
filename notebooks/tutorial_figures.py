"""
Tutorial-style figures — Bloomberg/FT explainer aesthetic.

Each figure addresses one concept. Big numbers, narrative subhead under the
title, callout boxes with the *why* (not just the what), reading guide on the
side, single focal point. Practitioner audience: a quant who needs to
understand the recipe in 60 seconds.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, "/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks")
from utils import garch_ar, split_cp_quantile  # noqa: E402

TUTORIAL_DIR = Path("/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/figures/tutorial")
WIKI_TUTORIAL_DIR = Path(
    "/media/ak/10E1026C4FA6006E/GitRepos/LLMWikiGeneration/wiki/wiki/assets/conformal-hft/tutorial"
)
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
WIKI_TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)

# FT-ish palette
INK = "#1d1d1b"
GRID = "#e6e6e6"
PRIMARY = "#0f5499"  # FT navy
ACCENT = "#cc0000"  # FT red
NEUTRAL = "#7a7a7a"
PAPER = "#fff1e5"  # FT salmon background
COVERED = "#066b35"
MISS = "#cc0000"
SOFT_BG = "#f6f6f6"


def setup_ft_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
        }
    )


def save_tutorial_fig(fig: plt.Figure, name: str) -> Path:
    local = TUTORIAL_DIR / name
    wiki = WIKI_TUTORIAL_DIR / name
    fig.savefig(local, facecolor=fig.get_facecolor())
    shutil.copyfile(local, wiki)
    return local


def headline(fig: plt.Figure, title: str, subhead: str, kicker: str = "CONFORMAL PREDICTION · STEP 2 OF 4") -> None:
    """FT-style title block: kicker, headline, subhead."""
    fig.text(0.06, 0.965, " ".join(list(kicker)), fontsize=9.5, color=ACCENT, fontweight="bold")
    fig.text(0.06, 0.93, title, fontsize=20, color=INK, fontweight="bold")
    fig.text(0.06, 0.895, subhead, fontsize=12.5, color=NEUTRAL, style="italic")
    fig.text(0.06, 0.04, "Source: synthetic AR(1)+GARCH(1,1) series · gradient-boosted base model · n_cal = 5,000",
             fontsize=8.5, color=NEUTRAL)


# =====================================================================
# Tutorial 02b — Where does q̂ come from?  (residual histogram, FT style)
# =====================================================================
def make_residual_quantile_ft(seed: int = 0, alpha: float = 0.1) -> None:
    setup_ft_style()
    series = garch_ar(T=20_000, seed=seed)
    X, y = series.X, series.y
    train = slice(0, 10_000)
    cal = slice(10_000, 15_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed)
    base.fit(X[train], y[train])
    abs_resid = np.abs(y[cal] - base.predict(X[cal]))
    q_hat = split_cp_quantile(y[cal] - base.predict(X[cal]), alpha=alpha)
    n_below = int(np.sum(abs_resid < q_hat))
    n_above = int(np.sum(abs_resid >= q_hat))
    pct_below = n_below / len(abs_resid)
    pct_above = n_above / len(abs_resid)

    fig = plt.figure(figsize=(14, 9.2))
    # Layout: headline (top), annotation strip, chart, source (bottom). Reading guide on right runs height of chart+strip.
    ax = fig.add_axes([0.07, 0.10, 0.58, 0.48])        # main histogram
    strip_ax = fig.add_axes([0.07, 0.62, 0.58, 0.18])  # annotation strip above chart
    guide_ax = fig.add_axes([0.69, 0.10, 0.27, 0.70])  # reading guide running full height

    # Histogram --------------------------------------------------------
    bins = np.linspace(0, abs_resid.max() * 1.02, 70)
    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(abs_resid, bins=bins)
    for i, c in enumerate(centers):
        color = PRIMARY if c < q_hat else ACCENT
        alpha_bar = 0.85 if c < q_hat else 0.65
        ax.bar(c, counts[i], width=(bins[1] - bins[0]) * 0.96, color=color, alpha=alpha_bar, edgecolor="white", lw=0.5)

    # q̂ vertical line + huge number
    ax.axvline(q_hat, color=INK, lw=2.2, zorder=5)
    y_top = counts.max() * 1.18
    ax.set_ylim(0, y_top)
    ax.set_xlim(0, abs_resid.max() * 1.02)
    # Big number for q̂
    ax.text(q_hat, y_top * 0.96, f"q̂ = {q_hat:.3f}", fontsize=22, fontweight="bold",
            color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=INK, lw=1.4))
    ax.text(q_hat + (abs_resid.max() * 0.018), y_top * 0.82,
            "the 90% quantile\nof |y − μ̂(x)|",
            fontsize=10.5, color=INK, ha="left", va="top", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8, alpha=0.95))

    # Dedicated annotation strip above the chart — never overlaps bars
    strip_ax.set_xlim(0, 1); strip_ax.set_ylim(0, 1)
    strip_ax.axis("off")
    # Left half: 90% block
    strip_ax.add_patch(Rectangle((0.0, 0.0), 0.48, 1.0,
                                 facecolor="#eaf2fb", edgecolor="none",
                                 transform=strip_ax.transAxes))
    strip_ax.text(0.02, 0.78, f"{pct_below:.0%}", fontsize=26, fontweight="bold",
                  color=PRIMARY, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.14, 0.74, "of past errors\nlive to the LEFT of q̂",
                  fontsize=11, fontweight="bold", color=PRIMARY,
                  ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.02, 0.30,
                  "These are the cases your 90% interval covers.\nThe band μ̂(x) ± q̂ wraps them all.",
                  fontsize=10, color=INK, ha="left", va="top", transform=strip_ax.transAxes)

    # Right half: 10% block
    strip_ax.add_patch(Rectangle((0.52, 0.0), 0.48, 1.0,
                                 facecolor="#fbeaea", edgecolor="none",
                                 transform=strip_ax.transAxes))
    strip_ax.text(0.54, 0.78, f"{pct_above:.0%}", fontsize=26, fontweight="bold",
                  color=ACCENT, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.62, 0.74, "live to the RIGHT of q̂",
                  fontsize=11, fontweight="bold", color=ACCENT,
                  ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.54, 0.30,
                  "The misses you accept by choosing α = 10%.\nNot a bug — your stated budget.",
                  fontsize=10, color=INK, ha="left", va="top", transform=strip_ax.transAxes)

    ax.set_xlabel("absolute residual on calibration set:  |y − μ̂(x)|", fontsize=11.5, labelpad=8)
    ax.set_ylabel("count of calibration points", fontsize=11.5)
    ax.tick_params(axis="both", labelsize=10)

    # Reading guide panel ---------------------------------------------
    guide_ax.set_xlim(0, 1); guide_ax.set_ylim(0, 1)
    guide_ax.axis("off")
    # Background card
    card = FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02",
                          fc=SOFT_BG, ec="#cccccc", lw=1, transform=guide_ax.transAxes)
    guide_ax.add_patch(card)
    guide_ax.text(0.5, 0.93, "H O W   T O   R E A D   T H I S", fontsize=10, fontweight="bold", color=ACCENT,
                  ha="center", va="top")
    guide_ax.text(0.07, 0.84, "1.  Bars = past prediction errors", fontsize=10.5, fontweight="bold", va="top")
    guide_ax.text(0.07, 0.79, "Each bar is one bucket of |y − μ̂|\nfrom the calibration set.",
                  fontsize=9.5, color="#444", va="top")
    guide_ax.text(0.07, 0.66, "2.  Black line = the 90% cut-off", fontsize=10.5, fontweight="bold", va="top")
    guide_ax.text(0.07, 0.61, "q̂ is the value below which 90% of\npast errors fall. Empirical, distribution-\nfree, finite-sample correct.",
                  fontsize=9.5, color="#444", va="top")
    guide_ax.text(0.07, 0.42, "3.  Your interval = μ̂(x) ± q̂", fontsize=10.5, fontweight="bold", va="top")
    guide_ax.text(0.07, 0.37, "Same q̂ for every test point. Under\nexchangeability of calibration and\ntest data, this band covers Y_test\nwith probability ≥ 90%.",
                  fontsize=9.5, color="#444", va="top")
    guide_ax.text(0.07, 0.16, "Why you should care", fontsize=10.5, fontweight="bold",
                  color=PRIMARY, va="top")
    guide_ax.text(0.07, 0.11, "No Gaussian assumption. Works on top\nof any base model. The 90% holds.",
                  fontsize=9.5, color="#444", va="top")

    # Headline block -----------------------------------------------
    headline(
        fig,
        title="Where does the conformal band's width come from?",
        subhead=("Take every error your model made on a held-out calibration set. "
                 "The 90% quantile of those errors is the half-width of your interval."),
        kicker="CONFORMAL PREDICTION · STEP 2 OF 4",
    )

    save_tutorial_fig(fig, "tutorial_02b_residuals_quantile_ft.png")
    plt.close(fig)
    print(f"wrote tutorial_02b_residuals_quantile_ft.png — q̂ = {q_hat:.4f}, "
          f"{pct_below:.1%} below / {pct_above:.1%} above")


if __name__ == "__main__":
    make_residual_quantile_ft()
