"""
Tutorial-style figures for the conformal-prediction-for-hft-traders wiki page.

These differ from the sim*_*.png in the parent folder: they are pedagogical,
not experimental. Each figure tells a story with annotations, callouts, and
semantic color. Moderate annotation density — 1-3 callouts per panel, pairs
with a 2-3 sentence caption in the markdown.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, "/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks")
from utils import garch_ar, setup_style, split_cp_quantile  # noqa: E402

TUTORIAL_DIR = Path("/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/figures/tutorial")
WIKI_TUTORIAL_DIR = Path(
    "/media/ak/10E1026C4FA6006E/GitRepos/LLMWikiGeneration/wiki/wiki/assets/conformal-hft/tutorial"
)
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
WIKI_TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)

COVERED = "#2ca02c"  # green
MISS = "#d62728"  # red
PRED = "#1f77b4"  # blue
BAND = "#7570b3"  # purple
HIGHLIGHT = "#ff7f0e"  # orange


def save_tutorial_fig(fig: plt.Figure, name: str) -> Path:
    local = TUTORIAL_DIR / name
    wiki = WIKI_TUTORIAL_DIR / name
    fig.savefig(local)
    shutil.copyfile(local, wiki)
    return local


# =====================================================================
# Tutorial figure 2: The split-CP recipe in 4 panels
# =====================================================================
def make_residual_quantile_figure(seed: int = 0, alpha: float = 0.1) -> None:
    setup_style()
    series = garch_ar(T=20_000, seed=seed)
    X, y = series.X, series.y
    train = slice(0, 10_000)
    cal = slice(10_000, 15_000)
    test = slice(15_000, 20_000)

    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed)
    base.fit(X[train], y[train])
    mu_cal = base.predict(X[cal])
    mu_test = base.predict(X[test])
    cal_resid = y[cal] - mu_cal
    abs_resid = np.abs(cal_resid)
    q_hat = split_cp_quantile(cal_resid, alpha=alpha)

    lo = mu_test - q_hat
    hi = mu_test + q_hat
    covered = (y[test] >= lo) & (y[test] <= hi)
    coverage_pct = covered.mean()
    n_test = len(y[test])

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    # Panel A: base model predictions on calibration set
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(y[cal], mu_cal, s=6, alpha=0.25, color=PRED, edgecolor="none")
    lims = [min(y[cal].min(), mu_cal.min()) * 1.05, max(y[cal].max(), mu_cal.max()) * 1.05]
    axA.plot(lims, lims, "--", color="black", lw=1, label="ideal y = μ̂")
    sample_idx = 17
    axA.annotate(
        f"residual\n= y - μ̂",
        xy=(y[cal][sample_idx], mu_cal[sample_idx]),
        xytext=(y[cal][sample_idx] - 0.6, mu_cal[sample_idx] + 0.6),
        arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=1.4),
        color=HIGHLIGHT,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    axA.scatter(
        [y[cal][sample_idx]], [mu_cal[sample_idx]],
        s=80, facecolor="none", edgecolor=HIGHLIGHT, lw=2, zorder=5,
    )
    axA.set_xlabel("realised y")
    axA.set_ylabel("model prediction μ̂(x)")
    axA.set_title("① Train a base model and predict on the calibration set", loc="left")
    axA.legend(loc="upper left", fontsize=9)

    # Panel B: histogram of residuals + quantile
    axB = fig.add_subplot(gs[0, 1])
    n, bins, patches = axB.hist(abs_resid, bins=60, color="lightsteelblue", edgecolor="white")
    # Color the bins below q_hat differently
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < q_hat:
            patch.set_facecolor("#a6cee3")
        else:
            patch.set_facecolor("#fb9a99")
    axB.axvline(q_hat, color=HIGHLIGHT, lw=2.2, label=f"q̂ = {q_hat:.3f}  (90% quantile)")
    axB.set_xlabel("|y − μ̂(x)|  (absolute residual on calibration set)")
    axB.set_ylabel("count")
    axB.set_title("② Take the 90% quantile of absolute residuals", loc="left")
    axB.annotate(
        "90% of residuals\nfall to the left of q̂",
        xy=(q_hat * 0.55, axB.get_ylim()[1] * 0.55),
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#a6cee3", lw=1.2),
    )
    axB.annotate(
        "10% to the right\n(the misses we'll accept)",
        xy=(q_hat * 1.4, axB.get_ylim()[1] * 0.35),
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#fb9a99", lw=1.2),
    )
    axB.legend(loc="upper right", fontsize=9)

    # Panel C: build band on test data, color hit/miss
    axC = fig.add_subplot(gs[1, 0])
    zoom = slice(0, 200)
    t_axis = np.arange(zoom.start, zoom.stop)
    axC.fill_between(
        t_axis, lo[zoom], hi[zoom],
        color=BAND, alpha=0.18, label=f"prediction band  μ̂ ± q̂",
    )
    axC.plot(t_axis, mu_test[zoom], color=PRED, lw=1.0, label="μ̂(x)")
    # Plot hits and misses
    hit_idx = np.where(covered[zoom])[0] + zoom.start
    miss_idx = np.where(~covered[zoom])[0] + zoom.start
    axC.scatter(hit_idx, y[test][hit_idx - zoom.start], s=10, color=COVERED,
                label=f"covered ({len(hit_idx)})", zorder=4, edgecolor="white", lw=0.3)
    axC.scatter(miss_idx, y[test][miss_idx - zoom.start], s=22, color=MISS,
                label=f"missed ({len(miss_idx)})", zorder=5, edgecolor="white", lw=0.5)
    if len(miss_idx) > 0:
        first_miss = miss_idx[0]
        axC.annotate(
            "miss",
            xy=(first_miss, y[test][first_miss - zoom.start]),
            xytext=(first_miss + 12, y[test][first_miss - zoom.start] + 0.4),
            arrowprops=dict(arrowstyle="->", color=MISS, lw=1.3),
            color=MISS, fontsize=10, fontweight="bold",
        )
    axC.set_xlabel("test index (first 200 shown)")
    axC.set_ylabel("y")
    axC.set_title("③ Apply band μ̂(x) ± q̂ to test data — green = covered, red = miss", loc="left")
    axC.legend(loc="upper right", fontsize=9, ncol=2)

    # Panel D: empirical coverage tally
    axD = fig.add_subplot(gs[1, 1])
    grid_n = 100  # show 100-point sample as a 10x10 grid
    rng = np.random.default_rng(seed)
    sample = rng.choice(n_test, size=grid_n, replace=False)
    grid_cov = covered[sample].reshape(10, 10)
    for i in range(10):
        for j in range(10):
            color = COVERED if grid_cov[i, j] else MISS
            axD.add_patch(plt.Rectangle((j, 9 - i), 0.92, 0.92, facecolor=color, edgecolor="white", lw=1.2))
    axD.set_xlim(-0.3, 10.3)
    axD.set_ylim(-3.2, 10.3)
    axD.set_aspect("equal")
    axD.axis("off")
    sample_cov = grid_cov.mean()
    full_cov = coverage_pct
    axD.text(5, -1.2, f"100-point sample: {grid_cov.sum()}/{grid_n} covered = {sample_cov:.2%}",
             ha="center", fontsize=10.5)
    axD.text(5, -2.2, f"Full test set ({n_test} pts):  {covered.sum()}/{n_test} covered = {full_cov:.2%}",
             ha="center", fontsize=11, fontweight="bold")
    on_target = abs(full_cov - (1 - alpha)) < 0.01
    axD.text(5, -3.0,
             f"target 1 − α = {1 - alpha:.0%}     achieved {full_cov:.1%}     {'✓ on target' if on_target else '~ near target'}",
             ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e8" if on_target else "#fff5e6",
                       ec=COVERED if on_target else HIGHLIGHT, lw=1.4))
    legend_handles = [
        mpatches.Patch(facecolor=COVERED, label="covered"),
        mpatches.Patch(facecolor=MISS, label="missed"),
    ]
    axD.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.05),
               ncol=2, fontsize=9.5, frameon=False)
    axD.set_title("④ Count covered points — empirical vs target", loc="left")

    fig.suptitle(
        "Split conformal prediction — the four-step recipe",
        fontsize=14, fontweight="bold", y=0.995,
    )
    save_tutorial_fig(fig, "tutorial_02_residual_quantile.png")
    plt.close(fig)
    print(f"wrote tutorial_02_residual_quantile.png — full coverage {full_cov:.4f}, q̂ = {q_hat:.4f}")


if __name__ == "__main__":
    make_residual_quantile_figure()
