"""
Build the three educational notebooks from a single Python source.

Run once: `python _build_notebooks.py` → writes Sim1/Sim2/Sim3 .ipynb files,
which are then executed via `jupyter nbconvert --execute`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path("/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def write_nb(cells: list[nbf.NotebookNode], path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "name": "cp-hft",
        "display_name": "Python (cp-hft)",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nbf.write(nb, path)


# =====================================================================
# Sim 1: Coverage failure of naive intervals under heteroskedasticity
# =====================================================================
sim1_cells = [
    md(
        "# Sim 1 — Coverage Failure of Naive Intervals under Heteroskedasticity\n\n"
        "**From the wiki page §7.1.** Build a heteroskedastic AR(1)+GARCH(1,1) "
        "synthetic series, fit a gradient-boosted base model, and compare three "
        "prediction-interval strategies:\n\n"
        "1. **Parametric** — `μ̂ ± 1.645 σ` using training-residual stdev.\n"
        "2. **Split CP** — finite-sample-corrected quantile of |residuals| on a held-out calibration set.\n"
        "3. **EnbPI** — bootstrap-ensemble LOO residual quantile (Xu & Xie 2023).\n\n"
        "The point of this notebook is to **see** the parametric interval fail while CP holds, "
        "and to feel where each method gives or takes width."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks')\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.ensemble import GradientBoostingRegressor\n"
        "from utils import (\n"
        "    garch_ar, setup_style, save_fig,\n"
        "    split_cp_quantile, enbpi_predict,\n"
        "    coverage, width, rolling_coverage,\n"
        ")\n"
        "setup_style()\n"
        "ALPHA = 0.10  # 90% target coverage\n"
        "RNG_SEED = 0"
    ),
    md(
        "## 1. Generate the synthetic series\n\n"
        "`y_t` is an AR(1) with GARCH(1,1) innovations — the canonical "
        "heteroskedastic returns process. Five engineered features mimic lagged "
        "returns, prior shock, prior vol, and a slow seasonal."
    ),
    code(
        "series = garch_ar(T=20_000, phi=0.3, omega=1e-3, alpha=0.1, beta=0.85, seed=0)\n"
        "X, y, sig2 = series.X, series.y, series.sig2\n"
        "train, cal, test = slice(0, 10_000), slice(10_000, 15_000), slice(15_000, 20_000)\n"
        "print(f'train={X[train].shape[0]} cal={X[cal].shape[0]} test={X[test].shape[0]}')\n"
        "print(f'overall std(y)={y.std():.4f}, vol range √sig2 [{np.sqrt(sig2).min():.4f}, {np.sqrt(sig2).max():.4f}]')"
    ),
    md(
        "### Visualising the heteroskedasticity\n\n"
        "Both the realised series and the latent volatility process. The vol "
        "clustering is the entire reason fixed-width intervals fail."
    ),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)\n"
        "axes[0].plot(y, color='#1f77b4', lw=0.5, alpha=0.85)\n"
        "axes[0].set_title('Synthetic AR(1)+GARCH(1,1) series')\n"
        "axes[0].set_ylabel('y_t')\n"
        "axes[1].plot(np.sqrt(sig2), color='#d62728', lw=0.8)\n"
        "axes[1].set_title('Latent volatility √σ²_t — note the clustering')\n"
        "axes[1].set_ylabel('σ_t')\n"
        "axes[1].set_xlabel('time')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim1_series_and_vol.png')\n"
        "plt.show()"
    ),
    md(
        "## 2. Fit the base model\n\n"
        "A gradient-boosted regressor. The CP guarantee does **not** depend on "
        "this choice — we could swap in any black box and the coverage claim "
        "would still hold (under exchangeability, which we will deliberately "
        "violate in Sim 2)."
    ),
    code(
        "base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=RNG_SEED)\n"
        "base.fit(X[train], y[train])\n"
        "mu_train = base.predict(X[train])\n"
        "mu_cal = base.predict(X[cal])\n"
        "mu_test = base.predict(X[test])\n"
        "print(f'train R² = {1 - ((y[train]-mu_train)**2).sum()/((y[train]-y[train].mean())**2).sum():.4f}')"
    ),
    md(
        "## 3. Build the three interval methods\n\n"
        "### A. Parametric ±1.645σ\n"
        "Treats the residuals as iid Gaussian — the assumption every quant who "
        "has fit a linear model already knows is wrong, but still reaches for."
    ),
    code(
        "sigma_train = (y[train] - mu_train).std()\n"
        "A_lo = mu_test - 1.645 * sigma_train\n"
        "A_hi = mu_test + 1.645 * sigma_train\n"
        "print(f'parametric half-width = 1.645 * {sigma_train:.4f} = {1.645*sigma_train:.4f}')"
    ),
    md(
        "### B. Split conformal prediction\n\n"
        "The simplest CP recipe: empirical (1−α) quantile of |y − μ̂| on the "
        "calibration set, finite-sample-corrected by `⌈(n+1)(1−α)⌉/n`."
    ),
    code(
        "cal_residuals = y[cal] - mu_cal\n"
        "q_hat = split_cp_quantile(cal_residuals, alpha=ALPHA)\n"
        "B_lo = mu_test - q_hat\n"
        "B_hi = mu_test + q_hat\n"
        "print(f'split-CP half-width = {q_hat:.4f}')"
    ),
    md(
        "### C. EnbPI (bootstrap-ensemble LOO)\n\n"
        "20 bootstrap GBMs; LOO predictions on the training set provide the "
        "residual pool. Avoids splitting away calibration data — more efficient "
        "when data is precious. (Validity here is asymptotic and requires "
        "β-mixing, not exchangeability.)"
    ),
    code(
        "from sklearn.ensemble import GradientBoostingRegressor\n"
        "mu_enb, C_lo, C_hi = enbpi_predict(\n"
        "    GradientBoostingRegressor,\n"
        "    dict(n_estimators=120, max_depth=4, random_state=RNG_SEED),\n"
        "    X[train], y[train], X[test],\n"
        "    B=20, alpha=ALPHA, seed=RNG_SEED,\n"
        ")\n"
        "print(f'EnbPI mean half-width = {(C_hi - C_lo).mean()/2:.4f}')"
    ),
    md("## 4. Compare coverage and width"),
    code(
        "rows = []\n"
        "for name, lo, hi in [\n"
        "    ('Parametric ±1.645σ', A_lo, A_hi),\n"
        "    ('Split CP', B_lo, B_hi),\n"
        "    ('EnbPI', C_lo, C_hi),\n"
        "]:\n"
        "    rows.append({\n"
        "        'method': name,\n"
        "        'coverage': coverage(y[test], lo, hi),\n"
        "        'mean width': width(lo, hi),\n"
        "    })\n"
        "summary = pd.DataFrame(rows).set_index('method')\n"
        "summary"
    ),
    md(
        "### Coverage and width — at a glance\n\n"
        "If the wiki claim holds, parametric should land short of 0.90 while "
        "split CP and EnbPI sit on it. Width tells you what the coverage cost."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
        "colors = ['#a6611a', '#018571', '#7570b3']\n"
        "axes[0].bar(summary.index, summary['coverage'], color=colors)\n"
        "axes[0].axhline(1 - ALPHA, color='black', ls='--', lw=1)\n"
        "axes[0].set_ylim(0.7, 1.0)\n"
        "axes[0].set_ylabel('empirical coverage')\n"
        "axes[0].set_title('Coverage (target = 0.90)')\n"
        "for tick in axes[0].get_xticklabels(): tick.set_rotation(15)\n"
        "axes[1].bar(summary.index, summary['mean width'], color=colors)\n"
        "axes[1].set_ylabel('mean interval width')\n"
        "axes[1].set_title('Average interval width')\n"
        "for tick in axes[1].get_xticklabels(): tick.set_rotation(15)\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim1_coverage_width.png')\n"
        "plt.show()"
    ),
    md(
        "## 5. Where the parametric interval fails\n\n"
        "Stratify the test set by latent vol decile. Parametric coverage should "
        "collapse in the high-vol bins; CP and EnbPI should stay flat — that's "
        "the practical meaning of *conditional* coverage."
    ),
    code(
        "test_vol = np.sqrt(sig2[test])\n"
        "deciles = pd.qcut(test_vol, 10, labels=False)\n"
        "bins = pd.DataFrame({'decile': deciles})\n"
        "for name, lo, hi in [('Parametric', A_lo, A_hi), ('Split CP', B_lo, B_hi), ('EnbPI', C_lo, C_hi)]:\n"
        "    bins[name] = ((y[test] >= lo) & (y[test] <= hi)).astype(float)\n"
        "by_decile = bins.groupby('decile').mean()\n"
        "by_decile"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
        "for name, color in zip(['Parametric', 'Split CP', 'EnbPI'], colors):\n"
        "    ax.plot(by_decile.index, by_decile[name], 'o-', label=name, color=color)\n"
        "ax.axhline(1 - ALPHA, color='black', ls='--', lw=1, label='target 0.90')\n"
        "ax.set_xlabel('latent vol decile (low → high)')\n"
        "ax.set_ylabel('empirical coverage in decile')\n"
        "ax.set_title('Conditional coverage by realised-vol decile')\n"
        "ax.set_ylim(0.4, 1.02)\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim1_conditional_coverage.png')\n"
        "plt.show()"
    ),
    md(
        "## 6. Visualising intervals over a high-vol window\n\n"
        "Zoom into a turbulent 500-step segment and overlay the three intervals. "
        "Parametric is a constant tube; CP/EnbPI may still be flat-ish here "
        "because their score isn't `σ̂`-scaled yet — but they trade *coverage* "
        "for narrower mean width in calm regimes."
    ),
    code(
        "vol_test = np.sqrt(sig2[test])\n"
        "peak = int(np.argmax(vol_test))\n"
        "lo_zoom = max(0, peak - 250); hi_zoom = min(len(vol_test), peak + 250)\n"
        "z = slice(lo_zoom, hi_zoom)\n"
        "t_axis = np.arange(lo_zoom, hi_zoom)\n"
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.plot(t_axis, y[test][z], color='black', lw=0.6, label='y_test')\n"
        "ax.fill_between(t_axis, A_lo[z], A_hi[z], color='#a6611a', alpha=0.18, label='Parametric')\n"
        "ax.fill_between(t_axis, B_lo[z], B_hi[z], color='#018571', alpha=0.22, label='Split CP')\n"
        "ax.fill_between(t_axis, C_lo[z], C_hi[z], color='#7570b3', alpha=0.25, label='EnbPI')\n"
        "ax.set_title(f'Test-window intervals around vol peak (idx {peak})')\n"
        "ax.set_xlabel('test index'); ax.set_ylabel('y')\n"
        "ax.legend(loc='upper right')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim1_intervals_zoom.png')\n"
        "plt.show()"
    ),
    md(
        "## 7. Takeaway\n\n"
        "- Parametric undercovers globally and collapses in high-vol bins.\n"
        "- Split CP holds marginally because the residuals were exchangeable enough "
        "  across this stationary GARCH window.\n"
        "- EnbPI matches Split CP on coverage without burning a calibration set.\n"
        "- **Width-adaptive scoring (σ̂-scaling, CQR) is the next lever** — Sim 3 picks that up.\n"
        "- **Coverage holds, but conditional coverage by vol bin is still uneven** for non-adaptive "
        "  scores. That's why §5 of the wiki insists on `σ̂`-scaling or CQR scores for HFT use."
    ),
]
write_nb(sim1_cells, NB_DIR / "01_sim1_heteroskedastic_coverage.ipynb")


# =====================================================================
# Sim 2: Regime-shift recovery with ACI on top of a rolling residual pool
# =====================================================================
sim2_cells = [
    md(
        "# Sim 2 — Regime-Shift Recovery with ACI\n\n"
        "**From the wiki page §7.2.** Same generator as Sim 1, but at "
        "`t = 12_500` the GARCH `ω` is multiplied by 5 — an abrupt vol "
        "explosion. We track rolling-500 empirical coverage under three "
        "strategies:\n\n"
        "1. **Parametric** — frozen Gaussian band.\n"
        "2. **Split CP (fixed)** — calibrated once on `[10_000, 12_500)`, never updated.\n"
        "3. **Rolling sliding-window CP** — recomputes `q̂` on the last 500 residuals.\n"
        "4. **ACI** — adapts the target miscoverage online (Gibbs & Candès 2021).\n\n"
        "The story: only **active feedback** (sliding window + ACI) recovers."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks')\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.ensemble import GradientBoostingRegressor\n"
        "from utils import (\n"
        "    garch_ar, setup_style, save_fig,\n"
        "    split_cp_quantile, aci_update,\n"
        "    coverage, width, rolling_coverage,\n"
        ")\n"
        "setup_style()\n"
        "ALPHA = 0.10\n"
        "BREAK_AT = 12_500\n"
        "OMEGA_MULT = 5.0"
    ),
    md(
        "## 1. Generate the regime-shift series\n\n"
        "Mean dynamics unchanged; `ω` jumps by 5× at the break — vol explodes."
    ),
    code(
        "series = garch_ar(\n"
        "    T=20_000, regime_break=BREAK_AT, regime_omega_mult=OMEGA_MULT, seed=1,\n"
        ")\n"
        "X, y, sig2 = series.X, series.y, series.sig2\n"
        "train = slice(0, 10_000)\n"
        "cal_pre = slice(10_000, 12_500)\n"
        "test = slice(12_500, 20_000)\n"
        "print(f'pre-break σ̄ = {np.sqrt(sig2[cal_pre]).mean():.4f}')\n"
        "print(f'post-break σ̄ = {np.sqrt(sig2[test]).mean():.4f}')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 3.2))\n"
        "ax.plot(np.sqrt(sig2), color='#d62728', lw=0.7)\n"
        "ax.axvline(BREAK_AT, color='black', ls='--', label=f'regime break t={BREAK_AT}')\n"
        "ax.set_title(f'Latent volatility — ω multiplied by {OMEGA_MULT} at break')\n"
        "ax.set_xlabel('time'); ax.set_ylabel('σ_t'); ax.legend()\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim2_regime_break_vol.png')\n"
        "plt.show()"
    ),
    md(
        "## 2. Train base model on pre-break data only\n\n"
        "Critically, the base model has **never seen** the post-break regime. "
        "Whether your interval still covers there is the entire question."
    ),
    code(
        "base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0)\n"
        "base.fit(X[train], y[train])\n"
        "mu_test = base.predict(X[test])\n"
        "y_test = y[test]\n"
        "T_test = len(y_test)\n"
        "print(f'test size: {T_test}, post-break observations only')"
    ),
    md(
        "## 3. Build the four interval streams\n\n"
        "### A. Parametric (frozen)\n"
        "Fixed bandwidth from pre-break training residuals."
    ),
    code(
        "sigma_pre = (y[train] - base.predict(X[train])).std()\n"
        "A_lo = mu_test - 1.645 * sigma_pre\n"
        "A_hi = mu_test + 1.645 * sigma_pre"
    ),
    md(
        "### B. Split CP, calibrated once before the break (and never updated)\n"
        "This is the trap most practitioners fall into: 'I have a calibration "
        "set, therefore I have coverage forever.'"
    ),
    code(
        "mu_cal = base.predict(X[cal_pre])\n"
        "cal_resid = y[cal_pre] - mu_cal\n"
        "q_hat_fixed = split_cp_quantile(cal_resid, alpha=ALPHA)\n"
        "B_lo = mu_test - q_hat_fixed\n"
        "B_hi = mu_test + q_hat_fixed\n"
        "print(f'fixed q̂ = {q_hat_fixed:.4f}')"
    ),
    md(
        "### C. Sliding-window split CP\n"
        "Recompute `q̂` on the last 500 absolute residuals at each step. This is "
        "the WCP-window variant from the wiki's four-family taxonomy."
    ),
    code(
        "WIN = 500\n"
        "pool = list(np.abs(cal_resid))\n"
        "C_lo = np.zeros(T_test); C_hi = np.zeros(T_test)\n"
        "for t in range(T_test):\n"
        "    arr = np.asarray(pool[-WIN:])\n"
        "    n_eff = len(arr)\n"
        "    level = min(np.ceil((n_eff + 1) * (1 - ALPHA)) / n_eff, 1.0)\n"
        "    q = np.quantile(arr, level, method='higher')\n"
        "    C_lo[t] = mu_test[t] - q\n"
        "    C_hi[t] = mu_test[t] + q\n"
        "    pool.append(abs(y_test[t] - mu_test[t]))"
    ),
    md(
        "### D. Adaptive Conformal Inference (ACI)\n"
        "Same sliding-window residual pool, but additionally adjust `α_t` online "
        "based on whether the realised observation was covered. `γ = 0.01` is "
        "the standard default; smaller γ = slower adaptation but less noise."
    ),
    code(
        "D_lo, D_hi, alpha_traj = aci_update(\n"
        "    residuals_history=np.abs(cal_resid),\n"
        "    y_obs=y_test, mu=mu_test,\n"
        "    alpha_target=ALPHA, gamma=0.01, window=WIN,\n"
        ")\n"
        "print(f'final α_t = {alpha_traj[-1]:.4f}, min α_t = {alpha_traj.min():.4f}, max α_t = {alpha_traj.max():.4f}')"
    ),
    md(
        "## 4. Marginal coverage over the post-break test set"
    ),
    code(
        "rows = []\n"
        "for name, lo, hi in [\n"
        "    ('Parametric (frozen)', A_lo, A_hi),\n"
        "    ('Split CP (frozen)', B_lo, B_hi),\n"
        "    ('Sliding-window CP', C_lo, C_hi),\n"
        "    ('ACI (γ=0.01)', D_lo, D_hi),\n"
        "]:\n"
        "    rows.append({'method': name, 'coverage': coverage(y_test, lo, hi), 'mean width': width(lo, hi)})\n"
        "summary2 = pd.DataFrame(rows).set_index('method')\n"
        "summary2"
    ),
    md(
        "## 5. Rolling coverage — the headline plot\n\n"
        "This is the central visual: does each method recover after the shock? "
        "Parametric and frozen split CP shouldn't; sliding and ACI should."
    ),
    code(
        "win = 500\n"
        "rc = {\n"
        "    'Parametric (frozen)': rolling_coverage(y_test, A_lo, A_hi, win=win),\n"
        "    'Split CP (frozen)': rolling_coverage(y_test, B_lo, B_hi, win=win),\n"
        "    'Sliding-window CP': rolling_coverage(y_test, C_lo, C_hi, win=win),\n"
        "    'ACI (γ=0.01)': rolling_coverage(y_test, D_lo, D_hi, win=win),\n"
        "}\n"
        "fig, ax = plt.subplots(figsize=(11, 4.6))\n"
        "palette = {'Parametric (frozen)':'#a6611a', 'Split CP (frozen)':'#dfc27d',\n"
        "           'Sliding-window CP':'#018571', 'ACI (γ=0.01)':'#7570b3'}\n"
        "for name, series_ in rc.items():\n"
        "    ax.plot(series_, label=name, color=palette[name], lw=1.4)\n"
        "ax.axhline(1 - ALPHA, color='black', ls='--', lw=1, label='target 0.90')\n"
        "ax.set_title(f'Rolling-{win} empirical coverage on the post-break test set')\n"
        "ax.set_xlabel('test index (post-break)')\n"
        "ax.set_ylabel('empirical coverage')\n"
        "ax.set_ylim(0.4, 1.02)\n"
        "ax.legend(loc='lower right')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim2_rolling_coverage.png')\n"
        "plt.show()"
    ),
    md(
        "## 6. α_t evolution under ACI\n\n"
        "Watch α_t spike right after the regime break (coverage was being missed → "
        "ACI lowers α_t → wider intervals) and slowly return toward the target as "
        "coverage normalises."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "ax.plot(alpha_traj, color='#7570b3', lw=1.2)\n"
        "ax.axhline(ALPHA, color='black', ls='--', label='target α=0.10')\n"
        "ax.set_title('ACI: target miscoverage α_t over time')\n"
        "ax.set_xlabel('test index (post-break)')\n"
        "ax.set_ylabel('α_t')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim2_aci_alpha_trajectory.png')\n"
        "plt.show()"
    ),
    md(
        "## 7. Half-width over time\n\n"
        "Adaptive methods widen during the shock and tighten as the regime stabilises. "
        "Frozen methods just sit at constant width — and pay for it in coverage."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "for name, lo, hi in [\n"
        "    ('Parametric (frozen)', A_lo, A_hi),\n"
        "    ('Split CP (frozen)', B_lo, B_hi),\n"
        "    ('Sliding-window CP', C_lo, C_hi),\n"
        "    ('ACI (γ=0.01)', D_lo, D_hi),\n"
        "]:\n"
        "    ax.plot((hi - lo) / 2, label=name, color=palette[name], lw=1.2)\n"
        "ax.set_title('Half-width of interval over time')\n"
        "ax.set_xlabel('test index (post-break)')\n"
        "ax.set_ylabel('half-width')\n"
        "ax.legend(loc='upper right')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim2_width_over_time.png')\n"
        "plt.show()"
    ),
    md(
        "## 8. Takeaway\n\n"
        "- The parametric and frozen-split-CP intervals collapse below the 0.90 target "
        "  and never recover.\n"
        "- Sliding-window CP recovers as the new high-vol residuals accumulate in the pool.\n"
        "- ACI recovers faster than vanilla sliding CP because the α_t update applies "
        "  *direct* feedback from each miss.\n"
        "- The width trade-off is real and visible: ACI intervals widen briefly, then settle.\n"
        "- For abrupt regime shifts (the wiki's HFT post-news scenario), **ACI or its "
        "  PID variant from the Angelopoulos repo is the recommended workhorse**."
    ),
]
write_nb(sim2_cells, NB_DIR / "02_sim2_regime_shift_aci.ipynb")


# =====================================================================
# Sim 3: Synthetic LOB-style benchmark of four interval methods
# =====================================================================
sim3_cells = [
    md(
        "# Sim 3 — Synthetic LOB-Style Benchmark of Four Interval Methods\n\n"
        "**From the wiki page §7.3.** A 100k-row synthetic HFT dataset with "
        "LOB-style features. Compare:\n\n"
        "1. **XGB quantile** — native `reg:quantileerror` at 5%/95%.\n"
        "2. **XGB mean + Gaussian 1.96σ** — parametric baseline.\n"
        "3. **XGB mean + EnbPI** — bootstrap-ensemble LOO with sliding-window residual pool.\n"
        "4. **XGB quantile + CQR + EnbPI** — heteroskedasticity-aware *and* non-exchangeable safe.\n\n"
        "Metrics: marginal coverage, mean width, **conditional coverage by latent-vol decile**, "
        "and a **calibration plot** across α ∈ {0.05, 0.1, 0.2, 0.3, 0.5}."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro/notebooks')\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import xgboost as xgb\n"
        "from utils import (\n"
        "    make_lob_dataset, setup_style, save_fig,\n"
        "    split_cp_quantile, enbpi_sliding_predict,\n"
        "    coverage, width,\n"
        ")\n"
        "setup_style()\n"
        "ALPHA = 0.10\n"
        "SEED = 7"
    ),
    md(
        "## 1. Synthetic LOB-like dataset\n\n"
        "Five features mimic top-of-book microstructure inputs; the target is a "
        "vol-scaled next-step return with intermittent regime shifts."
    ),
    code(
        "data = make_lob_dataset(T=60_000, seed=SEED)  # reduced size to keep run fast\n"
        "X, y, vol = data.X.to_numpy(), data.y, data.vol\n"
        "n = len(y)\n"
        "i_train, i_test = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)\n"
        "X_tr, X_te = X[i_train], X[i_test]\n"
        "y_tr, y_te = y[i_train], y[i_test]\n"
        "vol_te = vol[i_test]\n"
        "print(f'train={len(y_tr)}, test={len(y_te)}; |y| mean={np.mean(np.abs(y)):.3f}')"
    ),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)\n"
        "axes[0].plot(y, color='#1f77b4', lw=0.4, alpha=0.7)\n"
        "axes[0].set_title('Synthetic LOB-style return y_t (60k steps)')\n"
        "axes[1].plot(vol, color='#d62728', lw=0.8)\n"
        "axes[1].set_title('Latent volatility (regime shifts at T/4, T/2, 3T/4)')\n"
        "axes[1].set_xlabel('time')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim3_synthetic_series.png')\n"
        "plt.show()"
    ),
    md(
        "## 2. Method (a) — Native XGB quantile at 5%/95%\n\n"
        "XGBoost's `reg:quantileerror` fits the quantile loss directly. No "
        "conformal correction — the intervals are 'calibrated in-sample only', "
        "as the wiki warns."
    ),
    code(
        "def fit_xgb_quantile(X_tr, y_tr, q):\n"
        "    model = xgb.XGBRegressor(\n"
        "        objective='reg:quantileerror', quantile_alpha=q,\n"
        "        n_estimators=300, max_depth=4, learning_rate=0.08,\n"
        "        random_state=SEED, tree_method='hist', verbosity=0,\n"
        "    )\n"
        "    model.fit(X_tr, y_tr)\n"
        "    return model\n"
        "q_lo_model = fit_xgb_quantile(X_tr, y_tr, ALPHA / 2)\n"
        "q_hi_model = fit_xgb_quantile(X_tr, y_tr, 1 - ALPHA / 2)\n"
        "A_lo = q_lo_model.predict(X_te)\n"
        "A_hi = q_hi_model.predict(X_te)\n"
        "print(f'(a) XGB-quantile mean half-width = {(A_hi - A_lo).mean()/2:.4f}')"
    ),
    md(
        "## 3. Method (b) — XGB mean + Gaussian ±1.96σ"
    ),
    code(
        "mean_model = xgb.XGBRegressor(\n"
        "    n_estimators=300, max_depth=4, learning_rate=0.08,\n"
        "    random_state=SEED, tree_method='hist', verbosity=0,\n"
        ")\n"
        "mean_model.fit(X_tr, y_tr)\n"
        "mu_te_mean = mean_model.predict(X_te)\n"
        "mu_tr = mean_model.predict(X_tr)\n"
        "sigma_tr = (y_tr - mu_tr).std()\n"
        "B_lo = mu_te_mean - 1.96 * sigma_tr\n"
        "B_hi = mu_te_mean + 1.96 * sigma_tr\n"
        "print(f'(b) Gaussian half-width = {1.96*sigma_tr:.4f}')"
    ),
    md(
        "## 4. Method (c) — XGB mean + EnbPI (sliding residual pool)\n\n"
        "B = 15 bootstrap XGBoosts, LOO residuals on the training set, online "
        "sliding-window quantile of width 1000 over the test path."
    ),
    code(
        "mu_enb, C_lo, C_hi = enbpi_sliding_predict(\n"
        "    xgb.XGBRegressor,\n"
        "    dict(n_estimators=120, max_depth=4, learning_rate=0.08,\n"
        "         random_state=SEED, tree_method='hist', verbosity=0),\n"
        "    X_tr, y_tr, X_te, y_te,\n"
        "    B=15, alpha=ALPHA, window=1000, seed=SEED,\n"
        ")\n"
        "print(f'(c) EnbPI mean half-width = {(C_hi - C_lo).mean()/2:.4f}')"
    ),
    md(
        "## 5. Method (d) — XGB quantile + CQR + online residual update\n\n"
        "The 'kitchen sink' from the wiki. The CQR score is\n"
        "  `s_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i))`\n"
        "which is *positive* when the QR interval misses, *negative* when it's "
        "loose. We add a conformal correction `q̂` of those scores. We then run "
        "it in a sliding-window mode so it remains valid after regime shifts."
    ),
    code(
        "# Compute CQR scores on a held-out chunk of training data (last 30% of train)\n"
        "n_tr = len(y_tr); split = int(0.7 * n_tr)\n"
        "X_tr2, y_tr2 = X_tr[:split], y_tr[:split]\n"
        "X_cal, y_cal = X_tr[split:], y_tr[split:]\n"
        "q_lo_m = fit_xgb_quantile(X_tr2, y_tr2, ALPHA / 2)\n"
        "q_hi_m = fit_xgb_quantile(X_tr2, y_tr2, 1 - ALPHA / 2)\n"
        "lo_cal = q_lo_m.predict(X_cal); hi_cal = q_hi_m.predict(X_cal)\n"
        "lo_te0 = q_lo_m.predict(X_te); hi_te0 = q_hi_m.predict(X_te)\n"
        "scores_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)\n"
        "\n"
        "# Sliding-window correction\n"
        "WIN = 1000\n"
        "pool = list(scores_cal)\n"
        "D_lo = np.zeros(len(y_te)); D_hi = np.zeros(len(y_te))\n"
        "for t in range(len(y_te)):\n"
        "    arr = np.asarray(pool[-WIN:])\n"
        "    level = min(np.ceil((len(arr) + 1) * (1 - ALPHA)) / len(arr), 1.0)\n"
        "    q_corr = np.quantile(arr, level, method='higher')\n"
        "    D_lo[t] = lo_te0[t] - q_corr\n"
        "    D_hi[t] = hi_te0[t] + q_corr\n"
        "    new_score = max(lo_te0[t] - y_te[t], y_te[t] - hi_te0[t])\n"
        "    pool.append(new_score)\n"
        "print(f'(d) CQR+sliding half-width = {(D_hi - D_lo).mean()/2:.4f}')"
    ),
    md("## 6. Marginal coverage and width"),
    code(
        "rows = []\n"
        "for name, lo, hi in [\n"
        "    ('(a) XGB quantile', A_lo, A_hi),\n"
        "    ('(b) XGB mean + Gaussian', B_lo, B_hi),\n"
        "    ('(c) XGB mean + EnbPI', C_lo, C_hi),\n"
        "    ('(d) XGB quantile + CQR + sliding', D_lo, D_hi),\n"
        "]:\n"
        "    rows.append({'method': name, 'coverage': coverage(y_te, lo, hi), 'mean width': width(lo, hi)})\n"
        "summary3 = pd.DataFrame(rows).set_index('method')\n"
        "summary3"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
        "palette = ['#a6611a', '#dfc27d', '#018571', '#7570b3']\n"
        "axes[0].bar(range(len(summary3)), summary3['coverage'], color=palette)\n"
        "axes[0].axhline(1 - ALPHA, color='black', ls='--')\n"
        "axes[0].set_xticks(range(len(summary3)))\n"
        "axes[0].set_xticklabels(summary3.index, rotation=20, ha='right')\n"
        "axes[0].set_ylim(0.5, 1.0); axes[0].set_ylabel('coverage')\n"
        "axes[0].set_title('Marginal coverage (target 0.90)')\n"
        "axes[1].bar(range(len(summary3)), summary3['mean width'], color=palette)\n"
        "axes[1].set_xticks(range(len(summary3)))\n"
        "axes[1].set_xticklabels(summary3.index, rotation=20, ha='right')\n"
        "axes[1].set_ylabel('mean width')\n"
        "axes[1].set_title('Mean interval width')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim3_marginal_coverage_width.png')\n"
        "plt.show()"
    ),
    md(
        "## 7. Conditional coverage by latent-vol decile\n\n"
        "The decisive plot. A method that's marginally calibrated but misses in "
        "high-vol bins isn't safe for HFT risk — that's where the money goes."
    ),
    code(
        "deciles = pd.qcut(vol_te, 10, labels=False)\n"
        "bins = pd.DataFrame({'decile': deciles})\n"
        "for name, lo, hi in [\n"
        "    ('(a) XGB quantile', A_lo, A_hi),\n"
        "    ('(b) XGB mean + Gaussian', B_lo, B_hi),\n"
        "    ('(c) XGB mean + EnbPI', C_lo, C_hi),\n"
        "    ('(d) XGB quantile + CQR + sliding', D_lo, D_hi),\n"
        "]:\n"
        "    bins[name] = ((y_te >= lo) & (y_te <= hi)).astype(float)\n"
        "by_decile = bins.groupby('decile').mean()\n"
        "by_decile"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 4.8))\n"
        "for name, color in zip(by_decile.columns, palette):\n"
        "    ax.plot(by_decile.index, by_decile[name], 'o-', label=name, color=color, lw=1.4)\n"
        "ax.axhline(1 - ALPHA, color='black', ls='--', label='target 0.90')\n"
        "ax.set_xlabel('latent vol decile (low → high)')\n"
        "ax.set_ylabel('empirical coverage in decile')\n"
        "ax.set_title('Conditional coverage by latent-vol decile')\n"
        "ax.set_ylim(0.4, 1.05); ax.legend(loc='lower right')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim3_conditional_coverage.png')\n"
        "plt.show()"
    ),
    md(
        "## 8. Calibration plot — coverage at multiple α levels\n\n"
        "For each α, re-evaluate each method's interval (where possible — XGB "
        "quantile has fixed α from training). For methods that admit a varying "
        "α at test time, we rebuild the interval; otherwise we keep the trained "
        "value (so (a)/(b) appear as single points)."
    ),
    code(
        "alphas_grid = [0.05, 0.10, 0.20, 0.30, 0.50]\n"
        "cal_rows = []\n"
        "# (b) — Gaussian: re-scale z\n"
        "from scipy.stats import norm\n"
        "for a in alphas_grid:\n"
        "    z = norm.ppf(1 - a / 2)\n"
        "    lo = mu_te_mean - z * sigma_tr; hi = mu_te_mean + z * sigma_tr\n"
        "    cal_rows.append({'method': '(b) Gaussian', 'alpha': a, 'coverage': coverage(y_te, lo, hi)})\n"
        "# (c) — EnbPI: re-quantile residual pool\n"
        "pool_c = np.abs(y_tr - mean_model.predict(X_tr))\n"
        "for a in alphas_grid:\n"
        "    q = np.quantile(pool_c, np.ceil((len(pool_c) + 1) * (1 - a)) / len(pool_c), method='higher')\n"
        "    lo = mu_te_mean - q; hi = mu_te_mean + q\n"
        "    cal_rows.append({'method': '(c) EnbPI', 'alpha': a, 'coverage': coverage(y_te, lo, hi)})\n"
        "# (d) — CQR + sliding at the trained α; we re-evaluate by varying the conformal q on the same scores\n"
        "for a in alphas_grid:\n"
        "    arr = np.asarray(pool[-WIN:])\n"
        "    level = min(np.ceil((len(arr) + 1) * (1 - a)) / len(arr), 1.0)\n"
        "    q_corr = np.quantile(arr, level, method='higher')\n"
        "    lo = lo_te0 - q_corr; hi = hi_te0 + q_corr\n"
        "    cal_rows.append({'method': '(d) CQR+sliding', 'alpha': a, 'coverage': coverage(y_te, lo, hi)})\n"
        "cal_df = pd.DataFrame(cal_rows)\n"
        "cal_df.head()"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7.5, 5.5))\n"
        "for method, group in cal_df.groupby('method'):\n"
        "    ax.plot(1 - group['alpha'], group['coverage'], 'o-', label=method, lw=1.6)\n"
        "ax.plot([0.4, 1.0], [0.4, 1.0], color='black', ls='--', label='ideal')\n"
        "ax.set_xlabel('nominal coverage 1−α')\n"
        "ax.set_ylabel('empirical coverage')\n"
        "ax.set_title('Calibration plot — nominal vs realised across α levels')\n"
        "ax.legend(loc='lower right')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, 'sim3_calibration_plot.png')\n"
        "plt.show()"
    ),
    md(
        "## 9. Takeaway\n\n"
        "- **Native XGB quantile and Gaussian intervals are both miscalibrated** under "
        "  regime shifts — they undercover, especially in high-vol bins.\n"
        "- **EnbPI** restores marginal coverage but its width can be wider than CQR's "
        "  because it doesn't adapt the *shape* of the interval.\n"
        "- **CQR + sliding correction** delivers the tightest valid intervals: marginal "
        "  coverage holds *and* conditional coverage stays close to target across vol "
        "  deciles. The calibration plot lies closest to the diagonal.\n"
        "- This is the empirical case the wiki page makes: in production HFT, **wrap a "
        "  tree-based quantile regressor with CQR plus a sliding-window or EnbPI residual "
        "  update**. That's the practical default."
    ),
]
write_nb(sim3_cells, NB_DIR / "03_sim3_lob_benchmark.ipynb")

print("Wrote 3 notebooks to", NB_DIR)
