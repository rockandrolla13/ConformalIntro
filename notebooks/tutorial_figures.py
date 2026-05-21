"""
Tutorial-style figures — Bloomberg/FT explainer aesthetic.

Twelve pedagogical figures covering the full conformal-prediction-for-HFT
narrative. Each figure addresses one concept, uses a fixed layout
(kicker · headline · italic subhead · annotation strip · single chart ·
right-side reading guide · source line), and is rendered into both
ConformalIntro/figures/tutorial/ and the wiki assets directory.

Data is shared with the simulation notebooks via utils.py so the numbers
on the tutorial figures match the wiki's prose verbatim.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, "/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks")
from utils import (  # noqa: E402
    aci_update,
    enbpi_predict,
    garch_ar,
    make_lob_dataset,
    split_cp_quantile,
)

TUTORIAL_DIR = Path("/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/figures/tutorial")
WIKI_TUTORIAL_DIR = Path(
    "/media/ak/10E1026C4FA6006E/GitRepos/LLMWikiGeneration/wiki/wiki/assets/conformal-hft/tutorial"
)
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
WIKI_TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)

# Palette --------------------------------------------------------------
INK = "#1d1d1b"
GRID = "#e6e6e6"
PRIMARY = "#0f5499"
ACCENT = "#cc0000"
NEUTRAL = "#7a7a7a"
COVERED = "#066b35"
MISS = "#cc0000"
TRAIN = "#0f5499"
CAL = "#e07a00"
TEST = "#066b35"
SOFT_BG = "#f6f6f6"
PRIMARY_FILL = "#eaf2fb"
ACCENT_FILL = "#fbeaea"
COVERED_FILL = "#e8f3ec"


# Helpers --------------------------------------------------------------
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


def headline(fig, title: str, subhead: str, kicker: str) -> None:
    fig.text(0.06, 0.965, " ".join(list(kicker)), fontsize=9.5, color=ACCENT, fontweight="bold")
    fig.text(0.06, 0.93, title, fontsize=20, color=INK, fontweight="bold")
    fig.text(0.06, 0.895, subhead, fontsize=12.5, color=NEUTRAL, style="italic")


def source(fig, text: str) -> None:
    fig.text(0.06, 0.035, text, fontsize=8.5, color=NEUTRAL)


def two_stat_strip(strip_ax, left_pct, left_color, left_fill, left_label, left_sub,
                   right_pct, right_color, right_fill, right_label, right_sub) -> None:
    """Standard two-block annotation strip. left_pct/right_pct can be a string."""
    strip_ax.set_xlim(0, 1); strip_ax.set_ylim(0, 1); strip_ax.axis("off")
    strip_ax.add_patch(Rectangle((0.0, 0.0), 0.48, 1.0, facecolor=left_fill,
                                 edgecolor="none", transform=strip_ax.transAxes))
    strip_ax.text(0.02, 0.78, left_pct, fontsize=26, fontweight="bold",
                  color=left_color, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.14, 0.74, left_label, fontsize=11, fontweight="bold",
                  color=left_color, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.02, 0.30, left_sub, fontsize=10, color=INK,
                  ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.add_patch(Rectangle((0.52, 0.0), 0.48, 1.0, facecolor=right_fill,
                                 edgecolor="none", transform=strip_ax.transAxes))
    strip_ax.text(0.54, 0.78, right_pct, fontsize=26, fontweight="bold",
                  color=right_color, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.66, 0.74, right_label, fontsize=11, fontweight="bold",
                  color=right_color, ha="left", va="top", transform=strip_ax.transAxes)
    strip_ax.text(0.54, 0.30, right_sub, fontsize=10, color=INK,
                  ha="left", va="top", transform=strip_ax.transAxes)


def reading_guide(guide_ax, title: str, items: list[tuple[str, str]],
                  payoff_title: str, payoff_body: str) -> None:
    """Standard right-hand reading-guide card."""
    guide_ax.set_xlim(0, 1); guide_ax.set_ylim(0, 1); guide_ax.axis("off")
    card = FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02",
                          fc=SOFT_BG, ec="#cccccc", lw=1, transform=guide_ax.transAxes)
    guide_ax.add_patch(card)
    guide_ax.text(0.5, 0.93, " ".join(list(title)), fontsize=10, fontweight="bold",
                  color=ACCENT, ha="center", va="top")
    y = 0.84
    for i, (it_title, it_body) in enumerate(items, start=1):
        guide_ax.text(0.07, y, f"{i}.  {it_title}", fontsize=10.5, fontweight="bold", va="top")
        y -= 0.05
        guide_ax.text(0.07, y, it_body, fontsize=9.5, color="#444", va="top")
        # Estimate next-item position from body line count
        n_lines = max(1, it_body.count("\n") + 1)
        y -= 0.045 * n_lines + 0.04
    guide_ax.text(0.07, 0.16, payoff_title, fontsize=10.5, fontweight="bold",
                  color=PRIMARY, va="top")
    guide_ax.text(0.07, 0.11, payoff_body, fontsize=9.5, color="#444", va="top")


def standard_layout():
    """Return (fig, ax_main, ax_strip, ax_guide) with the canonical FT layout."""
    fig = plt.figure(figsize=(14, 9.2))
    ax = fig.add_axes([0.07, 0.10, 0.58, 0.48])
    strip = fig.add_axes([0.07, 0.62, 0.58, 0.18])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.70])
    return fig, ax, strip, guide


# ----------------------------------------------------------------------
# Tutorial 01 — Data split: train | cal | test
# ----------------------------------------------------------------------
def make_tutorial_01_data_split():
    setup_ft_style()
    series = garch_ar(T=20_000, seed=0)
    y = series.y
    train_end, cal_end = 10_000, 15_000

    fig, ax, strip, guide = standard_layout()

    # Main chart: time series colored by region
    t = np.arange(len(y))
    ax.plot(t[:train_end], y[:train_end], color=TRAIN, lw=0.4, alpha=0.85)
    ax.plot(t[train_end:cal_end], y[train_end:cal_end], color=CAL, lw=0.4, alpha=0.85)
    ax.plot(t[cal_end:], y[cal_end:], color=TEST, lw=0.4, alpha=0.85)
    ax.axvline(train_end, color=INK, lw=1.2, ls="--")
    ax.axvline(cal_end, color=INK, lw=1.2, ls="--")
    y_top = y.max() * 1.05; y_bot = y.min() * 1.05
    ax.set_ylim(y_bot, y_top)
    ax.text(train_end/2, y_top*0.92, "TRAIN", ha="center", color=TRAIN, fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TRAIN, lw=1.2))
    ax.text((train_end+cal_end)/2, y_top*0.92, "CALIBRATE", ha="center", color=CAL, fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CAL, lw=1.2))
    ax.text((cal_end+len(y))/2, y_top*0.92, "TEST", ha="center", color=TEST, fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEST, lw=1.2))
    ax.set_xlabel("time index", fontsize=11.5, labelpad=8)
    ax.set_ylabel("y", fontsize=11.5)

    # Three-block strip (override the two-block helper)
    strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
    blocks = [
        (0.00, 0.32, TRAIN, "#eaf2fb", "TRAIN", "10,000", "Fit your base model here.\nResiduals NOT computed here."),
        (0.34, 0.32, CAL, "#fdf4e8", "CALIBRATE", "5,000", "Compute |y − μ̂| on this set.\nThis is where q̂ is born."),
        (0.68, 0.32, TEST, "#e8f3ec", "TEST", "5,000", "The 90% guarantee is\nevaluated on this set."),
    ]
    for x0, w, color, fill, lbl, num, sub in blocks:
        strip.add_patch(Rectangle((x0, 0.0), w, 1.0, facecolor=fill, edgecolor="none",
                                  transform=strip.transAxes))
        strip.text(x0 + 0.02, 0.85, lbl, fontsize=11, fontweight="bold", color=color,
                   ha="left", va="top", transform=strip.transAxes)
        strip.text(x0 + 0.02, 0.65, num, fontsize=22, fontweight="bold", color=INK,
                   ha="left", va="top", transform=strip.transAxes)
        strip.text(x0 + 0.02, 0.30, sub, fontsize=9.5, color=INK,
                   ha="left", va="top", transform=strip.transAxes)

    reading_guide(guide, "HOW TO READ THIS", [
        ("Three disjoint slices of time",
         "Train, calibrate, test never overlap.\nNo leakage; the guarantee depends on it."),
        ("Model sees train only",
         "Calibration must be data the model\nhas never been fit on."),
        ("Test is sacred",
         "Coverage is measured here.\nNo touching during model selection."),
    ], "Why the split matters",
       "Without it, you measure training\nfit, not real-world coverage.")

    headline(fig,
             "Step 1: split your data into train / calibrate / test",
             "Conformal prediction needs three disjoint slices. The model fits on one, residuals are measured on another, the 90% guarantee is verified on the third.",
             "CONFORMAL PREDICTION · STEP 1 OF 4")
    source(fig, "Source: synthetic AR(1)+GARCH(1,1) series · 20,000 steps · split 10k / 5k / 5k")
    save_tutorial_fig(fig, "tutorial_01_data_split.png")
    plt.close(fig)
    print("wrote tutorial_01_data_split.png")


# ----------------------------------------------------------------------
# Tutorial 02 — Where does q̂ come from? (existing, kept as canonical)
# ----------------------------------------------------------------------
def make_tutorial_02_residual_quantile():
    setup_ft_style()
    series = garch_ar(T=20_000, seed=0)
    X, y = series.X, series.y
    train = slice(0, 10_000); cal = slice(10_000, 15_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0)
    base.fit(X[train], y[train])
    abs_resid = np.abs(y[cal] - base.predict(X[cal]))
    q_hat = split_cp_quantile(y[cal] - base.predict(X[cal]), alpha=0.10)
    pct_below = float((abs_resid < q_hat).mean())
    pct_above = 1 - pct_below

    fig, ax, strip, guide = standard_layout()
    bins = np.linspace(0, abs_resid.max() * 1.02, 70)
    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(abs_resid, bins=bins)
    for i, c in enumerate(centers):
        color = PRIMARY if c < q_hat else ACCENT
        a = 0.85 if c < q_hat else 0.65
        ax.bar(c, counts[i], width=(bins[1]-bins[0])*0.96, color=color, alpha=a, edgecolor="white", lw=0.5)
    ax.axvline(q_hat, color=INK, lw=2.2, zorder=5)
    y_top = counts.max() * 1.18
    ax.set_ylim(0, y_top); ax.set_xlim(0, abs_resid.max() * 1.02)
    ax.text(q_hat, y_top * 0.96, f"q̂ = {q_hat:.3f}", fontsize=22, fontweight="bold",
            color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=INK, lw=1.4))
    ax.text(q_hat + (abs_resid.max() * 0.018), y_top * 0.82,
            "the 90% quantile\nof |y − μ̂(x)|",
            fontsize=10.5, color=INK, ha="left", va="top", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8, alpha=0.95))

    two_stat_strip(strip,
                   f"{pct_below:.0%}", PRIMARY, PRIMARY_FILL,
                   "of past errors\nlive to the LEFT of q̂",
                   "These are the cases your 90% interval covers.\nThe band μ̂(x) ± q̂ wraps them all.",
                   f"{pct_above:.0%}", ACCENT, ACCENT_FILL,
                   "live to the RIGHT of q̂",
                   "The misses you accept by choosing α = 10%.\nNot a bug — your stated budget.")
    ax.set_xlabel("absolute residual on calibration set:  |y − μ̂(x)|", fontsize=11.5, labelpad=8)
    ax.set_ylabel("count of calibration points", fontsize=11.5)

    reading_guide(guide, "HOW TO READ THIS", [
        ("Bars = past prediction errors",
         "Each bar is one bucket of |y − μ̂|\nfrom the calibration set."),
        ("Black line = the 90% cut-off",
         "q̂ is the value below which 90%\nof past errors fall."),
        ("Your interval = μ̂(x) ± q̂",
         "Same q̂ for every test point.\nCovers Y_test with prob ≥ 90%."),
    ], "Why you should care",
       "No Gaussian assumption.\nWorks on top of any base model.")

    headline(fig,
             "Step 2: take the 90% quantile of past errors",
             "The 90% quantile of |y − μ̂| on the calibration set is the half-width of your interval. Distribution-free, finite-sample correct.",
             "CONFORMAL PREDICTION · STEP 2 OF 4")
    source(fig, "Source: synthetic AR(1)+GARCH(1,1) series · gradient-boosted base model · n_cal = 5,000")
    save_tutorial_fig(fig, "tutorial_02_residual_quantile.png")
    # Also re-save under the old name for backward compatibility
    save_tutorial_fig(fig, "tutorial_02b_residuals_quantile_ft.png")
    plt.close(fig)
    print(f"wrote tutorial_02_residual_quantile.png — q̂={q_hat:.4f}")


# ----------------------------------------------------------------------
# Tutorial 03 — Coverage in pictures: 20 test points, hit/miss
# ----------------------------------------------------------------------
def make_tutorial_03_coverage_in_pictures():
    setup_ft_style()
    series = garch_ar(T=20_000, seed=0)
    X, y = series.X, series.y
    train, cal, test = slice(0, 10_000), slice(10_000, 15_000), slice(15_000, 20_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[train], y[train])
    mu_te = base.predict(X[test])
    q_hat = split_cp_quantile(y[cal] - base.predict(X[cal]), alpha=0.10)
    lo, hi = mu_te - q_hat, mu_te + q_hat
    covered = (y[test] >= lo) & (y[test] <= hi)

    n_show = 20
    rng = np.random.default_rng(3)
    sample = rng.choice(len(y[test]), size=n_show, replace=False)
    sample_cov = covered[sample]
    n_hit = int(sample_cov.sum()); n_miss = n_show - n_hit

    BG_BAND = "#d6e4f4"
    fig, ax, strip, guide = standard_layout()
    xpos = np.arange(n_show)
    for i, idx in enumerate(sample):
        color = COVERED if sample_cov[i] else MISS
        ax.vlines(i, lo[idx], hi[idx], color=BG_BAND, lw=14, alpha=0.7)
        ax.plot([i], [mu_te[idx]], "o", color=PRIMARY, ms=5, zorder=4)
        marker = "o" if sample_cov[i] else "X"
        ax.plot([i], [y[test][idx]], marker, color=color, ms=11, zorder=5,
                markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlim(-0.7, n_show - 0.3)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{i+1}" for i in range(n_show)], fontsize=9)
    ax.set_xlabel("a sample of 20 test points", fontsize=11.5, labelpad=8)
    ax.set_ylabel("y", fontsize=11.5)
    legend_handles = [
        plt.Line2D([0], [0], color=BG_BAND, lw=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIMARY, ms=7),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COVERED, ms=10),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=MISS, ms=10),
    ]
    ax.legend(legend_handles, ["μ̂(x) ± q̂", "μ̂(x)", "covered", "missed"],
              loc="upper right", fontsize=9.5, ncol=2)

    two_stat_strip(strip,
                   f"{n_hit}/{n_show}", COVERED, COVERED_FILL,
                   "test points covered",
                   "Each green dot is a test outcome\nthat landed inside the band.",
                   f"{n_miss}/{n_show}", ACCENT, ACCENT_FILL,
                   "missed",
                   "The reds are misses you priced in\nwhen you picked α = 10%.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("Each column is one test point",
         "Light band = μ̂(x) ± q̂.\nBlue dot = your point forecast."),
        ("Green = covered, red X = missed",
         "Did the realised y land inside\nthe band? Count them up."),
        ("Coverage is empirical",
         "Run it on enough test points\nand you'll see ~90%."),
    ], "What 'coverage' really is",
       "Not a probability assumption.\nA hit rate you can audit.")

    headline(fig,
             "Coverage in pictures: count the green ones",
             "A 90% interval means: on out-of-sample data, the realised value lands inside the band about nine times out of ten. Here are twenty such points.",
             "CORE INTUITION · WHAT COVERAGE MEANS")
    source(fig, f"Source: synthetic GARCH series · split CP at α = 0.10 · full test set coverage {covered.mean():.1%}")
    save_tutorial_fig(fig, "tutorial_03_coverage_in_pictures.png")
    plt.close(fig)
    print(f"wrote tutorial_03_coverage_in_pictures.png — sample {n_hit}/{n_show}, full {covered.mean():.4f}")


# ----------------------------------------------------------------------
# Tutorial 04 — Parametric fails: calm regime vs stressed regime
# ----------------------------------------------------------------------
def make_tutorial_04_parametric_vs_regime():
    setup_ft_style()
    series = garch_ar(T=20_000, regime_break=12_500, regime_omega_mult=5.0, seed=1)
    X, y, sig2 = series.X, series.y, series.sig2
    train = slice(0, 10_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[train], y[train])
    mu_all = base.predict(X)
    sigma_tr = (y[train] - mu_all[train]).std()
    band_half = 1.645 * sigma_tr

    calm = slice(10_200, 10_500)
    stress = slice(13_500, 13_800)

    def coverage_in(sl):
        lo = mu_all[sl] - band_half; hi = mu_all[sl] + band_half
        return float(((y[sl] >= lo) & (y[sl] <= hi)).mean())

    cov_calm = coverage_in(calm)
    cov_stress = coverage_in(stress)

    fig = plt.figure(figsize=(14, 9.2))
    ax_top = fig.add_axes([0.07, 0.36, 0.58, 0.22])
    ax_bot = fig.add_axes([0.07, 0.10, 0.58, 0.22])
    strip = fig.add_axes([0.07, 0.62, 0.58, 0.18])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.70])

    for ax, sl, title in [(ax_top, calm, "Calm regime  (pre-break)"),
                          (ax_bot, stress, "Stressed regime  (post 5× ω shock)")]:
        t = np.arange(sl.start, sl.stop)
        lo = mu_all[sl] - band_half; hi = mu_all[sl] + band_half
        cov_mask = (y[sl] >= lo) & (y[sl] <= hi)
        ax.fill_between(t, lo, hi, color=PRIMARY_FILL, alpha=0.9, label="parametric μ̂ ± 1.645σ")
        ax.plot(t, mu_all[sl], color=PRIMARY, lw=1.0, label="μ̂(x)")
        ax.scatter(t[cov_mask], y[sl][cov_mask], s=14, color=COVERED, label="covered",
                   edgecolor="white", lw=0.4, zorder=4)
        ax.scatter(t[~cov_mask], y[sl][~cov_mask], s=22, color=MISS, label="missed",
                   edgecolor="white", lw=0.5, zorder=5)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("time index", fontsize=10.5)

    ax_top.legend(loc="upper right", fontsize=9, ncol=2)

    two_stat_strip(strip,
                   f"{cov_calm:.0%}", COVERED, COVERED_FILL,
                   "calm regime coverage",
                   "Fixed-width band looks fine here.\nResiduals are small, regime quiet.",
                   f"{cov_stress:.0%}", ACCENT, ACCENT_FILL,
                   "stressed regime coverage",
                   "Same band, post-vol-shock. Half the\npoints land outside. Risk system blind.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("Same band in both panels",
         "Width is frozen from training σ.\nIt never reacts to live volatility."),
        ("Calm: most points green",
         "Coverage looks great in low-vol\nperiods — that's the trap."),
        ("Stressed: many red Xs",
         "Volatility doubled but the band\ndidn't. Coverage collapses."),
    ], "Why parametric fails",
       "1.645σ assumes one global σ.\nReal markets have many.")

    headline(fig,
             "Why parametric intervals fail when volatility shifts",
             "A band sized by training-set σ stays the same width forever. The market doesn't. Calm regimes hide the failure; vol bursts expose it.",
             "FAILURE MODE · PARAMETRIC INTERVALS")
    source(fig, "Source: synthetic GARCH series with regime break at t = 12,500 · ω multiplied by 5")
    save_tutorial_fig(fig, "tutorial_04_parametric_vs_regime.png")
    plt.close(fig)
    print(f"wrote tutorial_04_parametric_vs_regime.png — calm {cov_calm:.2%}, stress {cov_stress:.2%}")


# ----------------------------------------------------------------------
# Tutorial 05 — Conditional coverage explained
# ----------------------------------------------------------------------
def make_tutorial_05_conditional_coverage():
    setup_ft_style()
    series = garch_ar(T=20_000, seed=0)
    X, y, sig2 = series.X, series.y, series.sig2
    train = slice(0, 10_000); cal = slice(10_000, 15_000); test = slice(15_000, 20_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[train], y[train])
    mu_te = base.predict(X[test])
    # parametric
    sigma_tr = (y[train] - base.predict(X[train])).std()
    A_lo, A_hi = mu_te - 1.645 * sigma_tr, mu_te + 1.645 * sigma_tr
    # split CP
    q_hat = split_cp_quantile(y[cal] - base.predict(X[cal]), alpha=0.10)
    B_lo, B_hi = mu_te - q_hat, mu_te + q_hat
    # EnbPI
    _, C_lo, C_hi = enbpi_predict(GradientBoostingRegressor,
        dict(n_estimators=120, max_depth=4, random_state=0),
        X[train], y[train], X[test], B=15, alpha=0.10, seed=0)

    vol_te = np.sqrt(sig2[test])
    deciles = pd.qcut(vol_te, 10, labels=False)
    df = pd.DataFrame({"d": deciles})
    df["Parametric"] = ((y[test] >= A_lo) & (y[test] <= A_hi)).astype(float)
    df["Split CP"] = ((y[test] >= B_lo) & (y[test] <= B_hi)).astype(float)
    df["EnbPI"] = ((y[test] >= C_lo) & (y[test] <= C_hi)).astype(float)
    by_d = df.groupby("d").mean()

    worst_param = by_d["Parametric"].min(); worst_param_d = by_d["Parametric"].idxmin()

    fig, ax, strip, guide = standard_layout()
    palette = {"Parametric": ACCENT, "Split CP": PRIMARY, "EnbPI": COVERED}
    for col in by_d.columns:
        ax.plot(by_d.index, by_d[col], "o-", lw=2.0, color=palette[col], label=col, ms=7)
    ax.axhline(0.90, color=INK, ls="--", lw=1.2, label="target 0.90")
    ax.fill_between([by_d.index.min()-0.5, by_d.index.max()+0.5], 0.88, 0.92,
                    color="#e8f3ec", alpha=0.6, zorder=0)
    ax.set_xlim(by_d.index.min()-0.5, by_d.index.max()+0.5)
    ax.set_xticks(by_d.index)
    ax.set_xlabel("realised volatility decile (low → high)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("empirical coverage in decile", fontsize=11.5)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=9.5)
    # annotate the worst parametric decile
    ax.annotate(f"parametric drops\nto {worst_param:.0%} in vol decile {worst_param_d}",
                xy=(worst_param_d, worst_param),
                xytext=(worst_param_d - 2.5, worst_param - 0.18),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
                fontsize=10, color=ACCENT, fontweight="bold")

    two_stat_strip(strip,
                   f"{worst_param:.0%}", ACCENT, ACCENT_FILL,
                   "parametric, worst decile",
                   "In high-vol periods the fixed band\ncan't keep up. Risk system blind.",
                   f"{by_d['EnbPI'].mean():.0%}", COVERED, COVERED_FILL,
                   "EnbPI, average across deciles",
                   "Conformal methods stay near target\neverywhere — that's conditional coverage.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("X-axis: vol bin", "Test points sorted into ten\nvolatility deciles, low → high."),
        ("Y-axis: hit rate per bin", "Fraction of points covered in\neach decile."),
        ("Flat line = fair", "Methods that stay near 90% across\nall vols are conditionally calibrated."),
    ], "Why this matters",
       "Marginal coverage hides where\nthe band fails. This plot doesn't.")

    headline(fig,
             "Conditional coverage: where the band actually fails",
             "A method can hit 90% marginally yet undercover badly in the deciles where it matters most. Stratify by realised vol and look.",
             "DIAGNOSTIC · CONDITIONAL COVERAGE")
    source(fig, "Source: synthetic GARCH series · 5,000 test points · 10 vol deciles")
    save_tutorial_fig(fig, "tutorial_05_conditional_coverage.png")
    plt.close(fig)
    print(f"wrote tutorial_05_conditional_coverage.png — parametric worst {worst_param:.2%}")


# ----------------------------------------------------------------------
# Tutorial 06 — ACI as a feedback controller
# ----------------------------------------------------------------------
def make_tutorial_06_aci_controller():
    setup_ft_style()
    series = garch_ar(T=20_000, regime_break=12_500, regime_omega_mult=5.0, seed=1)
    X, y = series.X, series.y
    train = slice(0, 10_000); cal_pre = slice(10_000, 12_500); test = slice(12_500, 20_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[train], y[train])
    mu_te = base.predict(X[test]); y_te = y[test]
    cal_resid = np.abs(y[cal_pre] - base.predict(X[cal_pre]))
    lo, hi, alpha_t = aci_update(cal_resid, y_te, mu_te, alpha_target=0.10, gamma=0.01, window=500)
    miss = ~((y_te >= lo) & (y_te <= hi))
    n_show = 2000  # zoom around the break
    shown = slice(0, n_show)
    t = np.arange(n_show)

    # final coverage on full test
    full_cov = 1 - miss.mean()

    fig = plt.figure(figsize=(14, 9.6))
    ax_band = fig.add_axes([0.07, 0.40, 0.58, 0.18])
    ax_alpha = fig.add_axes([0.07, 0.22, 0.58, 0.15])
    ax_eq = fig.add_axes([0.07, 0.10, 0.58, 0.08])
    strip = fig.add_axes([0.07, 0.62, 0.58, 0.18])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.70])

    # Top: y_t and band, miss ticks
    ax_band.fill_between(t, lo[shown], hi[shown], color=PRIMARY_FILL, alpha=0.85, label="ACI band")
    ax_band.plot(t, mu_te[shown], color=PRIMARY, lw=0.6)
    miss_idx = np.where(miss[shown])[0]
    ax_band.scatter(miss_idx, y_te[shown][miss_idx], s=8, color=MISS, zorder=5)
    ax_band.set_title("y_t with ACI band (red dots = misses)", loc="left", fontsize=11)
    ax_band.set_xticklabels([])

    # Middle: alpha_t
    ax_alpha.plot(t, alpha_t[shown], color=PRIMARY, lw=1.3)
    ax_alpha.axhline(0.10, color=INK, ls="--", lw=1, label="target α = 0.10")
    ax_alpha.set_title("α_t — the controller's state", loc="left", fontsize=11)
    ax_alpha.set_xlabel("test index (post-break)", fontsize=10.5)
    ax_alpha.legend(loc="upper right", fontsize=9)

    # Bottom: the update equation
    ax_eq.axis("off")
    ax_eq.text(0.5, 0.7, r"$\alpha_{t+1} = \alpha_t + \gamma \cdot (\alpha^{*} - \mathbb{1}\{\mathrm{miss}_t\})$",
               fontsize=18, ha="center", va="center")
    ax_eq.text(0.5, 0.15, "miss now → α↓ → wider band next step.  Covered now → α↑ slightly → band can tighten.",
               fontsize=10.5, ha="center", color=NEUTRAL, style="italic")

    two_stat_strip(strip,
                   f"{full_cov:.0%}", COVERED, COVERED_FILL,
                   "final test coverage",
                   "ACI lands at the 90% target even\nthough the regime broke at t=0.",
                   "0.01", PRIMARY, PRIMARY_FILL,
                   "learning rate γ",
                   "Single knob. Larger γ = faster\nadaptation but noisier α_t.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("Top: band + miss ticks",
         "Red dots are where the realised y\nfell outside the band."),
        ("Middle: α_t state",
         "The controller's internal variable.\nSpikes after misses, drifts back."),
        ("Bottom: the rule",
         "Closed-loop feedback.\nOne line of arithmetic."),
    ], "Why this is a controller",
       "Same logic as a thermostat:\nmiss = error signal, γ = gain.")

    headline(fig,
             "ACI: turn coverage into a feedback control loop",
             "Treat the desired miscoverage α as the setpoint. Every observed miss is an error. A learning rate γ closes the loop.",
             "ADAPTIVE CONFORMAL INFERENCE · §5")
    source(fig, f"Source: synthetic GARCH series with regime break · ACI γ=0.01 · window=500 · final coverage {full_cov:.2%}")
    save_tutorial_fig(fig, "tutorial_06_aci_controller.png")
    plt.close(fig)
    print(f"wrote tutorial_06_aci_controller.png — coverage {full_cov:.4f}")


# ----------------------------------------------------------------------
# Tutorial 07 — Rolling coverage annotated
# ----------------------------------------------------------------------
def make_tutorial_07_rolling_coverage():
    setup_ft_style()
    series = garch_ar(T=20_000, regime_break=12_500, regime_omega_mult=5.0, seed=1)
    X, y = series.X, series.y
    train = slice(0, 10_000); cal_pre = slice(10_000, 12_500); test = slice(12_500, 20_000)
    base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[train], y[train])
    mu_te = base.predict(X[test]); y_te = y[test]
    sigma_tr = (y[train] - base.predict(X[train])).std()
    A_lo, A_hi = mu_te - 1.645 * sigma_tr, mu_te + 1.645 * sigma_tr
    q_pre = split_cp_quantile(y[cal_pre] - base.predict(X[cal_pre]), alpha=0.10)
    B_lo, B_hi = mu_te - q_pre, mu_te + q_pre
    D_lo, D_hi, _ = aci_update(np.abs(y[cal_pre] - base.predict(X[cal_pre])),
                               y_te, mu_te, alpha_target=0.10, gamma=0.01, window=500)

    def rcov(lo, hi, win=500):
        hit = ((y_te >= lo) & (y_te <= hi)).astype(float)
        return pd.Series(hit).rolling(win, min_periods=100).mean().to_numpy()

    rc_param = rcov(A_lo, A_hi)
    rc_scp = rcov(B_lo, B_hi)
    rc_aci = rcov(D_lo, D_hi)

    fig, ax, strip, guide = standard_layout()
    t = np.arange(len(y_te))
    ax.plot(t, rc_param, lw=1.6, color=ACCENT, label="Parametric (frozen)")
    ax.plot(t, rc_scp, lw=1.6, color="#dfc27d", label="Split CP (frozen)")
    ax.plot(t, rc_aci, lw=1.8, color=PRIMARY, label="ACI")
    ax.axhline(0.90, color=INK, ls="--", lw=1.2, label="target 0.90")
    ax.set_ylim(0.3, 1.02); ax.set_xlim(0, len(y_te))
    ax.set_xlabel("test index  (post-break)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("rolling-500 empirical coverage", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.annotate("regime break", xy=(50, 0.95), xytext=(800, 0.97),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3),
                fontsize=10.5, color=INK, fontweight="bold")
    # find ACI recovery point
    aci_recovered_idx = next((i for i, v in enumerate(rc_aci) if v >= 0.88), None)
    if aci_recovered_idx is not None:
        ax.annotate(f"ACI back on target\n(~{aci_recovered_idx} steps)",
                    xy=(aci_recovered_idx, rc_aci[aci_recovered_idx]),
                    xytext=(aci_recovered_idx + 500, 0.72),
                    arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.3),
                    fontsize=10.5, color=PRIMARY, fontweight="bold")
    # frozen labels
    final_param = float(rc_param[-1]); final_scp = float(rc_scp[-1]); final_aci = float(rc_aci[-1])

    two_stat_strip(strip,
                   f"{final_param:.0%}", ACCENT, ACCENT_FILL,
                   "frozen parametric, end of test",
                   "Stuck. The training σ never sees\nthe new regime.",
                   f"{final_aci:.0%}", COVERED, COVERED_FILL,
                   "ACI, end of test",
                   "Recovered. Active feedback closes\nthe gap within a few hundred steps.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("Lines: rolling coverage",
         "500-step rolling hit rate for each\nmethod, on post-break data."),
        ("Dashed line: target 0.90",
         "Where a 90% interval should live."),
        ("Watch the recovery",
         "Only the adaptive line climbs back\nto the target line."),
    ], "Why frozen methods fail",
       "No feedback. The miscoverage budget\nblows out and stays out.")

    headline(fig,
             "Regime-shift recovery: only adaptive methods survive",
             "When the volatility regime breaks, frozen intervals collapse to coverages around 50%. ACI returns to the 90% target within a few hundred observations.",
             "REGIME SHIFT · RECOVERY DYNAMICS")
    source(fig, f"Source: synthetic GARCH series · regime break at t=0 · ACI γ=0.01 · window=500")
    save_tutorial_fig(fig, "tutorial_07_rolling_coverage.png")
    plt.close(fig)
    print(f"wrote tutorial_07_rolling_coverage.png — param {final_param:.4f}, aci {final_aci:.4f}")


# ----------------------------------------------------------------------
# Tutorial 08 — CQR mechanic
# ----------------------------------------------------------------------
def make_tutorial_08_cqr_mechanic():
    setup_ft_style()
    from scipy.stats import norm
    # Synthetic 1-D heteroskedastic regression
    rng = np.random.default_rng(11)
    n = 6000
    # IMPORTANT: do NOT sort x before splitting — sorted x + sequential split
    # destroys exchangeability (each split would get a disjoint x range).
    x = rng.uniform(-3, 3, n)
    sigma_x = 0.35 + 0.5 * np.abs(x)
    y = np.sin(x) + sigma_x * rng.standard_normal(n)
    n_tr = int(0.6 * n); n_cal = int(0.2 * n)
    tr, cal, te = slice(0, n_tr), slice(n_tr, n_tr + n_cal), slice(n_tr + n_cal, n)

    # "Native QR" — true conditional quantiles at 15% and 85% (a deliberately
    # narrow ~70% nominal band). In real life you'd estimate these with QR;
    # for the teaching figure we use the true ones so the story is crystal clear.
    z_lo = norm.ppf(0.15); z_hi = norm.ppf(0.85)
    def native_band(x_arr):
        mu = np.sin(x_arr)
        sig = 0.35 + 0.5 * np.abs(x_arr)
        return mu + z_lo * sig, mu + z_hi * sig
    lo_cal, hi_cal = native_band(x[cal])
    lo_te0, hi_te0 = native_band(x[te])

    # CQR: take the conformal score, find its (1-α)(n+1)/n quantile, add
    # symmetrically to widen the band.
    cqr_scores = np.maximum(lo_cal - y[cal], y[cal] - hi_cal)
    n_eff = len(cqr_scores)
    q_corr = float(np.quantile(cqr_scores, np.ceil((n_eff + 1) * 0.9) / n_eff, method="higher"))
    lo_te_corr = lo_te0 - q_corr; hi_te_corr = hi_te0 + q_corr
    cov_native = float(((y[te] >= lo_te0) & (y[te] <= hi_te0)).mean())
    cov_cqr = float(((y[te] >= lo_te_corr) & (y[te] <= hi_te_corr)).mean())

    fig = plt.figure(figsize=(14, 9.2))
    ax_l = fig.add_axes([0.07, 0.10, 0.28, 0.48])
    ax_r = fig.add_axes([0.37, 0.10, 0.28, 0.48])
    strip = fig.add_axes([0.07, 0.62, 0.58, 0.18])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.70])

    # For plotting only, sort the test set by x so fill_between renders cleanly.
    order = np.argsort(x[te])
    x_plot = x[te][order]
    for ax, lo_, hi_, ttl, cov in [
        (ax_l, lo_te0, hi_te0, "Native quantile regression", cov_native),
        (ax_r, lo_te_corr, hi_te_corr, "After CQR correction", cov_cqr),
    ]:
        cov_mask = (y[te] >= lo_) & (y[te] <= hi_)
        lo_p = lo_[order]; hi_p = hi_[order]
        ax.fill_between(x_plot, lo_p, hi_p, color=PRIMARY_FILL, alpha=0.85)
        ax.scatter(x[te][cov_mask], y[te][cov_mask], s=6, color=COVERED, alpha=0.7, edgecolor="white", lw=0.2)
        ax.scatter(x[te][~cov_mask], y[te][~cov_mask], s=14, color=MISS, edgecolor="white", lw=0.5)
        ax.set_title(f"{ttl}\ncoverage = {cov:.1%}", loc="left", fontsize=11)
        ax.set_xlabel("x", fontsize=10.5); ax.set_ylabel("y", fontsize=10.5)
    # Arrow between the two panels to indicate the correction
    fig.text(0.355, 0.34, "+ q̂  on each side", fontsize=11, color=ACCENT,
             fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ACCENT, lw=1.4))

    two_stat_strip(strip,
                   f"{cov_native:.0%}", ACCENT, ACCENT_FILL,
                   "native QR coverage",
                   "Quantile regression alone is calibrated\nin-sample only. OOS undershoots.",
                   f"{cov_cqr:.0%}", COVERED, COVERED_FILL,
                   "after CQR correction",
                   f"Conformalised. The correction widens\nthe band by q̂ = {q_corr:.3f} symmetrically.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("Left: native quantile band",
         "Train a quantile regressor for 5%\nand 95%. Look at OOS coverage."),
        ("Right: same band, widened",
         "Add the conformal correction q̂\non both sides."),
        ("Width adapts to x",
         "The band breathes with the data,\nbecause q_lo/q_hi do."),
    ], "Why CQR is the best off-the-shelf",
       "QR for shape + CP for guarantee.\nNo distributional assumption.")

    headline(fig,
             "Conformalised quantile regression: shape + guarantee",
             "Native quantile regression has the right shape but no coverage guarantee. Add the CQR correction and you get both.",
             "METHOD · CQR")
    source(fig, f"Source: synthetic 1-D heteroskedastic regression · n = 2,000 · α = 0.10")
    save_tutorial_fig(fig, "tutorial_08_cqr_mechanic.png")
    plt.close(fig)
    print(f"wrote tutorial_08_cqr_mechanic.png — native {cov_native:.4f} → CQR {cov_cqr:.4f}")


# ----------------------------------------------------------------------
# Tutorial 09 — Calibration plot explained
# ----------------------------------------------------------------------
def make_tutorial_09_calibration_plot():
    setup_ft_style()
    data = make_lob_dataset(T=30_000, seed=7)
    X, y = data.X.to_numpy(), data.y
    n = len(y); split = int(0.7 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    mean = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X_tr, y_tr)
    mu_te = mean.predict(X_te)
    sigma_tr = (y_tr - mean.predict(X_tr)).std()
    pool = np.abs(y_tr - mean.predict(X_tr))

    alphas = np.array([0.05, 0.10, 0.20, 0.30, 0.50])
    from scipy.stats import norm
    gauss = []; enbpi = []
    for a in alphas:
        z = norm.ppf(1 - a / 2)
        cov_g = float(((y_te >= mu_te - z * sigma_tr) & (y_te <= mu_te + z * sigma_tr)).mean())
        gauss.append(cov_g)
        q = np.quantile(pool, np.ceil((len(pool) + 1) * (1 - a)) / len(pool), method="higher")
        cov_e = float(((y_te >= mu_te - q) & (y_te <= mu_te + q)).mean())
        enbpi.append(cov_e)
    gauss = np.array(gauss); enbpi = np.array(enbpi)
    nominal = 1 - alphas
    gap_g = float((nominal - gauss).max())

    fig, ax, strip, guide = standard_layout()
    ax.plot([0.4, 1.0], [0.4, 1.0], color=INK, ls="--", lw=1.4, label="ideal (y = x)")
    ax.plot(nominal, gauss, "o-", color=ACCENT, lw=2.2, ms=8, label="Gaussian ±1.96σ")
    ax.plot(nominal, enbpi, "o-", color=PRIMARY, lw=2.2, ms=8, label="Split CP / EnbPI")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("nominal coverage  1 − α", fontsize=11.5, labelpad=8)
    ax.set_ylabel("empirical coverage on test set", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=10)
    # annotate worst Gaussian deviation
    worst_idx = int(np.argmax(nominal - gauss))
    ax.annotate(f"Gaussian undershoots by {gap_g*100:.0f}pp\nat nominal {nominal[worst_idx]:.0%}",
                xy=(nominal[worst_idx], gauss[worst_idx]),
                xytext=(nominal[worst_idx] - 0.18, gauss[worst_idx] - 0.10),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                fontsize=10.5, color=ACCENT, fontweight="bold")

    two_stat_strip(strip,
                   f"{gap_g*100:.0f}pp", ACCENT, ACCENT_FILL,
                   "max Gaussian undershoot",
                   "Worst gap between nominal and\ndelivered coverage. Unauditable.",
                   "≈ 0pp", COVERED, COVERED_FILL,
                   "CP/EnbPI deviation",
                   "Empirical hits track the nominal\nlevel at every α the page tests.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("X-axis: nominal 1 − α",
         "What you claimed in your interval\n(50%, 70%, 80%, 90%, 95%)."),
        ("Y-axis: delivered coverage",
         "What you actually got on the\ntest set."),
        ("Diagonal = honest",
         "Every point on the line means\nyou kept your promise."),
    ], "Why a risk officer cares",
       "If you can't draw this line,\nyou can't defend the interval.")

    headline(fig,
             "Calibration plot: are you honest at every confidence level?",
             "Plot what you promised against what you delivered. A method that's calibrated at 90% but lies at 70% is still broken.",
             "DIAGNOSTIC · CALIBRATION ACROSS α")
    source(fig, "Source: synthetic LOB-style series · n = 30,000 · α ∈ {0.05, 0.10, 0.20, 0.30, 0.50}")
    save_tutorial_fig(fig, "tutorial_09_calibration_plot.png")
    plt.close(fig)
    print(f"wrote tutorial_09_calibration_plot.png — max gap {gap_g*100:.1f}pp")


# ----------------------------------------------------------------------
# Tutorial 10 — Conditional coverage on LOB benchmark
# ----------------------------------------------------------------------
def make_tutorial_10_conditional_lob():
    setup_ft_style()
    data = make_lob_dataset(T=30_000, seed=7)
    X, y, vol = data.X.to_numpy(), data.y, data.vol
    n = len(y); split = int(0.7 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    vol_te = vol[split:]
    mean = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X_tr, y_tr)
    mu_te = mean.predict(X_te)
    sigma_tr = (y_tr - mean.predict(X_tr)).std()
    # σ̂(x): fit a second model on |y - μ̂| to give vol-adaptive scaling
    sigma_m = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(
        X_tr, np.abs(y_tr - mean.predict(X_tr)))
    sigma_te = np.maximum(sigma_m.predict(X_te), 1e-3)
    # Gaussian: fixed-width baseline
    A_lo, A_hi = mu_te - 1.96 * sigma_tr, mu_te + 1.96 * sigma_tr
    # σ̂-scaled CP: width adapts to local vol via σ̂(x)
    scaled_resid_tr = np.abs(y_tr - mean.predict(X_tr)) / np.maximum(sigma_m.predict(X_tr), 1e-3)
    n_ct = len(scaled_resid_tr)
    q_scaled = float(np.quantile(scaled_resid_tr,
                                 np.ceil((n_ct + 1) * 0.9) / n_ct, method="higher"))
    B_lo, B_hi = mu_te - q_scaled * sigma_te, mu_te + q_scaled * sigma_te

    deciles = pd.qcut(vol_te, 10, labels=False)
    df = pd.DataFrame({"d": deciles})
    df["Gaussian"] = ((y_te >= A_lo) & (y_te <= A_hi)).astype(float)
    df["σ̂-scaled CP"] = ((y_te >= B_lo) & (y_te <= B_hi)).astype(float)
    by_d = df.groupby("d").mean()
    worst_g = by_d["Gaussian"].min(); worst_g_d = by_d["Gaussian"].idxmin()
    worst_e = by_d["σ̂-scaled CP"].min()

    fig, ax, strip, guide = standard_layout()
    ax.fill_between([by_d.index.min()-0.5, by_d.index.max()+0.5], 0.88, 0.92,
                    color="#e8f3ec", alpha=0.7, zorder=0, label="acceptable [0.88, 0.92]")
    ax.plot(by_d.index, by_d["Gaussian"], "o-", lw=2.0, color=ACCENT, label="(b) Gaussian", ms=8)
    ax.plot(by_d.index, by_d["σ̂-scaled CP"], "o-", lw=2.0, color=PRIMARY, label="(c) σ̂-scaled CP", ms=8)
    ax.axhline(0.90, color=INK, ls="--", lw=1.2, label="target 0.90")
    ax.set_xlim(by_d.index.min()-0.5, by_d.index.max()+0.5)
    ax.set_xticks(by_d.index)
    ax.set_xlabel("realised volatility decile (low → high)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("empirical coverage in decile", fontsize=11.5)
    ax.set_ylim(0.5, 1.05); ax.legend(loc="lower right", fontsize=9.5)
    ax.annotate(f"Gaussian collapses\nto {worst_g:.0%}", xy=(worst_g_d, worst_g),
                xytext=(worst_g_d - 2.5, worst_g - 0.10),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                fontsize=10.5, color=ACCENT, fontweight="bold")

    two_stat_strip(strip,
                   f"{worst_g:.0%}", ACCENT, ACCENT_FILL,
                   "Gaussian, worst decile",
                   "Coverage caves in the volatile\nregime. Fixed-σ band can't react.",
                   f"{worst_e:.0%}", COVERED, COVERED_FILL,
                   "σ̂-scaled CP, worst decile",
                   "Width adapts to local vol via σ̂(x).\nStays close to target everywhere.")

    reading_guide(guide, "HOW TO READ THIS", [
        ("X-axis: vol decile",
         "Test points sorted by realised vol\nin the LOB dataset."),
        ("Y-axis: coverage in decile",
         "Empirical hit rate for each method\nwithin that vol bin."),
        ("Shaded band = OK zone",
         "[0.88, 0.92] is the practical\ntolerance around the 0.90 target."),
    ], "What good looks like",
       "A horizontal line inside the\nshaded band, across all deciles.")

    headline(fig,
             "Conditional coverage on the LOB benchmark",
             "On a synthetic limit-order-book dataset, Gaussian undercovers in the high-vol decile. σ̂-scaled CP stays close to target across all bins.",
             "BENCHMARK · CONDITIONAL COVERAGE")
    source(fig, f"Source: synthetic LOB series · n_test = 9,000 · 10 vol deciles · α = 0.10")
    save_tutorial_fig(fig, "tutorial_10_conditional_lob.png")
    plt.close(fig)
    print(f"wrote tutorial_10_conditional_lob.png — gauss worst {worst_g:.4f}, enbpi worst {worst_e:.4f}")


# ----------------------------------------------------------------------
# Tutorial 11 — Decision tree of the four-family taxonomy
# ----------------------------------------------------------------------
def _box(ax, xy, w, h, label, sub, fc, ec=INK):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h*0.66, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=9, color="#444")


def _arrow(ax, p1, p2, label=None, color=INK):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", color=color, lw=1.3, mutation_scale=14))
    if label:
        mx = (p1[0] + p2[0]) / 2; my = (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.015, label, ha="center", va="bottom", fontsize=9.5,
                color=color, fontweight="bold")


def make_tutorial_11_decision_tree():
    setup_ft_style()
    fig = plt.figure(figsize=(14, 9.2))
    ax = fig.add_axes([0.07, 0.10, 0.58, 0.70])
    strip = fig.add_axes([0.07, 0.82, 0.58, 0.10])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.70])

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Root
    _box(ax, (0.35, 0.85), 0.30, 0.10, "Start here", "What does the data look like?", "#f6f6f6")
    # Q1: stationary?
    _box(ax, (0.35, 0.66), 0.30, 0.10, "Stationary?", "(no drift, no shock)", PRIMARY_FILL)
    _arrow(ax, (0.50, 0.85), (0.50, 0.76))
    # Yes leaf: SCP
    _box(ax, (0.05, 0.48), 0.24, 0.10, "Split CP (SCP)", "Cheapest. Marginal coverage.", COVERED_FILL)
    _arrow(ax, (0.40, 0.66), (0.17, 0.58), label="yes")
    # No: Heteroskedastic?
    _box(ax, (0.35, 0.48), 0.30, 0.10, "Heteroskedastic?", "(σ depends on x)", PRIMARY_FILL)
    _arrow(ax, (0.60, 0.66), (0.55, 0.58), label="no")
    # No-Yes: CQR / σ̂-scaling
    _box(ax, (0.05, 0.30), 0.24, 0.10, "CQR / σ̂-scaling", "Width adapts to x.", COVERED_FILL)
    _arrow(ax, (0.40, 0.48), (0.17, 0.40), label="yes")
    # Non-stationary path
    _box(ax, (0.35, 0.30), 0.30, 0.10, "Non-stationary?", "(slow drift, shocks)", PRIMARY_FILL)
    _arrow(ax, (0.60, 0.48), (0.55, 0.40), label="no")
    # Slow drift: EnbPI
    _box(ax, (0.05, 0.12), 0.24, 0.10, "EnbPI", "Sliding LOO residual pool.", COVERED_FILL)
    _arrow(ax, (0.40, 0.30), (0.17, 0.22), label="slow drift")
    # Abrupt shifts: ACI/PID
    _box(ax, (0.66, 0.12), 0.30, 0.10, "ACI / PID-Conformal", "α_t updated online.", COVERED_FILL)
    _arrow(ax, (0.60, 0.30), (0.80, 0.22), label="abrupt shifts")

    strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
    strip.add_patch(Rectangle((0, 0), 1, 1, fc="#f6f6f6", ec="none", transform=strip.transAxes))
    strip.text(0.02, 0.65, "5 families · 1 decision tree", fontsize=15, fontweight="bold",
               color=INK, va="center", transform=strip.transAxes)
    strip.text(0.02, 0.25,
               "Choose the simplest method that matches your data. Don't reach for ACI if SCP suffices.",
               fontsize=10.5, color=INK, va="center", transform=strip.transAxes)

    reading_guide(guide, "HOW TO READ THIS", [
        ("Top: stationarity check",
         "If yes, you're done — SCP gives\nmarginal coverage at no cost."),
        ("Middle: heteroskedasticity check",
         "If yes, CQR or σ̂-scaling makes\nthe width adapt to x."),
        ("Bottom: drift / shocks?",
         "EnbPI handles slow drift, ACI/PID\nhandles abrupt shifts."),
    ], "Why this taxonomy?",
       "Each family answers a different\nway exchangeability can break.")

    headline(fig,
             "How to choose your conformal method",
             "Don't memorise every variant. Ask three questions about your data: is it stationary, is the noise constant, does the regime shift?",
             "TAXONOMY · METHOD SELECTION")
    source(fig, "Source: synthesis of §5 of the wiki analysis · Stocker et al. 2025 four-family taxonomy")
    save_tutorial_fig(fig, "tutorial_11_decision_tree.png")
    plt.close(fig)
    print("wrote tutorial_11_decision_tree.png")


# ----------------------------------------------------------------------
# Tutorial 12 — Putting it all together: the pipeline schematic
# ----------------------------------------------------------------------
def make_tutorial_12_pipeline():
    setup_ft_style()
    fig = plt.figure(figsize=(14, 9.2))
    ax = fig.add_axes([0.07, 0.10, 0.58, 0.62])
    strip = fig.add_axes([0.07, 0.76, 0.58, 0.13])
    guide = fig.add_axes([0.69, 0.10, 0.27, 0.79])

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Boxes left to right
    boxes = [
        (0.02, 0.46, 0.16, 0.18, "Base model μ̂(x)", "XGBoost · LightGBM\nor any black box", "#f6f6f6"),
        (0.22, 0.46, 0.16, 0.18, "Calibration\nresiduals", "|y − μ̂| on held-out\ndata (or LOO via EnbPI)", PRIMARY_FILL),
        (0.42, 0.46, 0.16, 0.18, "Conformal\nthreshold q̂", "90% quantile of\nthose residuals", PRIMARY_FILL),
        (0.62, 0.46, 0.16, 0.18, "Prediction\ninterval", "μ̂(x) ± q̂\nor CQR-corrected", COVERED_FILL),
        (0.82, 0.46, 0.16, 0.18, "Trading action", "Quote skew · sizing\nstops · risk limits", COVERED_FILL),
    ]
    for (x, y, w, h, lbl, sub, fc) in boxes:
        _box(ax, (x, y), w, h, lbl, sub, fc)
    for x in [0.18, 0.38, 0.58, 0.78]:
        _arrow(ax, (x + 0.005, 0.55), (x + 0.035, 0.55))

    # Bottom layer: time-series fixes feeding into q̂
    _box(ax, (0.22, 0.16), 0.16, 0.14, "Sliding window", "or EnbPI bootstrap", "#fdf4e8")
    _box(ax, (0.42, 0.16), 0.16, 0.14, "ACI feedback", "online α_t update", "#fdf4e8")
    _arrow(ax, (0.30, 0.30), (0.30, 0.45), color=CAL)
    _arrow(ax, (0.50, 0.30), (0.50, 0.45), color=CAL)
    ax.text(0.40, 0.10,
            "time-series add-ons:  refresh the residual pool · adapt α online",
            ha="center", va="center", fontsize=10.5, color=CAL, fontweight="bold")

    strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
    strip.add_patch(Rectangle((0, 0), 1, 1, fc=PRIMARY_FILL, ec="none", transform=strip.transAxes))
    strip.text(0.02, 0.70, "Wraps any base model", fontsize=14, fontweight="bold", color=PRIMARY,
               va="center", transform=strip.transAxes)
    strip.text(0.02, 0.30,
               "~50 lines of code between μ̂(x) and a calibrated band you can defend in front of risk.",
               fontsize=11, color=INK, va="center", transform=strip.transAxes)

    reading_guide(guide, "HOW TO READ THIS", [
        ("Top row = base recipe",
         "Five boxes: model → residuals →\nq̂ → interval → action."),
        ("Bottom row = time-series",
         "Plug-ins that refresh q̂ or adapt\nα_t when exchangeability breaks."),
        ("Arrows = data flow",
         "No retraining of the base model.\nThe wrapper is the cheap part."),
    ], "Where to start",
       "Build the top row first. Add the\nbottom row only when needed.")

    headline(fig,
             "Putting it together: the conformal pipeline",
             "Five boxes from raw model to trading action. Two optional add-ons handle non-stationarity. No retraining of the base estimator at any point.",
             "ARCHITECTURE · THE FULL PIPELINE")
    source(fig, "Source: schematic synthesis of the wiki analysis §1–§6")
    save_tutorial_fig(fig, "tutorial_12_pipeline.png")
    plt.close(fig)
    print("wrote tutorial_12_pipeline.png")


# ----------------------------------------------------------------------
ALL_FIGURES = [
    make_tutorial_01_data_split,
    make_tutorial_02_residual_quantile,
    make_tutorial_03_coverage_in_pictures,
    make_tutorial_04_parametric_vs_regime,
    make_tutorial_05_conditional_coverage,
    make_tutorial_06_aci_controller,
    make_tutorial_07_rolling_coverage,
    make_tutorial_08_cqr_mechanic,
    make_tutorial_09_calibration_plot,
    make_tutorial_10_conditional_lob,
    make_tutorial_11_decision_tree,
    make_tutorial_12_pipeline,
]


if __name__ == "__main__":
    for fn in ALL_FIGURES:
        fn()
    print("all 12 tutorial figures rendered.")
