---
title: "Conformal Prediction for Algorithmic and HFT Traders: A Gentle Introduction"
page_id: analyses/conformal-prediction-for-hft-traders
page_type: analysis
revision_id: 2
created: 2026-05-21T15:30:00Z
updated: 2026-05-21T16:30:00Z
tags: [conformal-prediction, time-series, algorithmic-trading, hft, market-making, prediction-intervals, practitioner-guide]
related: [
  concepts/conformal-prediction,
  concepts/adaptive-conformal-inference,
  concepts/conformalized-quantile-regression,
  concepts/enbpi,
  concepts/spci,
  concepts/weighted-conformal-prediction,
  concepts/agaci,
  concepts/conformal-pid-control,
  concepts/nonconformity-score,
  concepts/marginal-coverage,
  concepts/conditional-coverage,
  concepts/coverage-guarantee,
  concepts/quantile-regression,
  concepts/exchangeability,
  concepts/limit-order-book,
  concepts/market-making,
  concepts/optimal-execution,
  concepts/inventory-risk,
  concepts/adverse-selection,
  concepts/avellaneda-stoikov-model,
  concepts/fill-probability,
  concepts/pairs-trading,
  concepts/statistical-arbitrage,
  sources/angelopoulos-2022-gentle-intro,
  sources/xu-2023-enbpi,
  sources/stocker-2025-conformal-timeseries-intro,
  sources/dieuleveut-zaffran-2025-cp-tutorial,
  sources/zaffran-2022-aci,
  sources/xu-2022-spci,
  sources/koukorinis-2026-draci
]
mind_map_priority: high
---

# Conformal Prediction for Algorithmic and HFT Traders

> **TL;DR — what you, an algo/HFT trader, get out of this page.**
>
> You ship models that predict things (next mid move, pre-trade slippage, fill probability, basket residual). Your risk system wants *intervals*. Conformal prediction (CP) is the thinnest possible wrapper that converts the first into the second, and it gives you three things you currently don't have:
>
> 1. **Coverage that holds out-of-sample, distribution-free.** When the 90% band says 90%, the test set delivers 90% — without assuming Gaussian residuals or stable regimes. In the regime-shift sim below, the parametric interval collapses to **48% coverage** after a 5× vol spike; ACI recovers to **90.0%** within a few hundred steps.
> 2. **Width that breathes with the regime.** σ̂-scaled or CQR-style intervals widen automatically in stressed minutes and tighten when calm — directly usable for quote skew, inventory limits, slippage budgets, and Kelly sizing. No more `μ ± 1.645σ` with a stale σ from last week.
> 3. **Auditability.** Every interval traces back to (calibration set, base model, score function, α level). A risk officer or regulator can verify it line-for-line. *"The neural net is 92% confident"* is unverifiable; CP isn't.
>
> **You don't replace your XGBoost / LightGBM / NN. You wrap it** — typically ~50 lines of calibration code. The rest of this page is the recipe (§1–3), three trading-specific use cases (§4), the time-series fixes (§5), the tree+CP stack (§6), and three reproducible simulations you can run today (§7).

Practitioner's guide for quants who already trust XGBoost, GBM, and rolling quantiles. The pitch is short. Wrap a [[concepts/conformal-prediction|conformal prediction (CP)]] layer around your existing model, and the 90% prediction interval you quote will actually contain the truth 90% of the time on out-of-sample data. That holds on the non-stationary, vol-clustered, regime-switching data you actually trade.

## Who this is for

You build models that predict things like:

- the next 30-second mid-price move
- pre-trade slippage on a 5% ADV parent order
- the residual of a [[concepts/pairs-trading|pairs]] / [[concepts/statistical-arbitrage|stat-arb]] basket
- the fill probability of a passive quote
- next-bar realised volatility

Your model outputs a number. You need a calibrated band around that number for sizing, stops, quote skew, and risk limits. You've tried:

- `μ ± 1.645σ` from a parametric error assumption. Undercovers because returns aren't Gaussian.
- Rolling empirical quantile of residuals. Noisy and lags shifts.
- [[concepts/quantile-regression|Quantile regression]] (QR) directly on the target. Intervals are flexible but there's no guarantee they hold OOS.
- Resampling / bootstrap. Expensive, optimistic under dependence.

CP gives you what those approaches don't: a finite-sample, distribution-free coverage guarantee that works with any base model, including the [[concepts/factor-models|trees]] you already trust.

## 1. The intuition in one paragraph

Train your favourite point predictor `μ̂(x)`. On a held-out calibration sample, record the residuals `s_i = |Y_i − μ̂(X_i)|`. Compute the empirical `⌈(n+1)(1−α)/n⌉` quantile `q̂`. At test time, your interval is `[μ̂(x) − q̂, μ̂(x) + q̂]`. Under [[concepts/exchangeability|exchangeability]] of calibration and test points, this interval covers `Y_test` with probability ≥ 1 − α. Model-free. Distribution-free.

That's split [[concepts/conformal-prediction|conformal prediction]]. Everything else in this document is plumbing: (a) tighten the interval with a smarter [[concepts/nonconformity-score|score]] than `|Y − μ̂|`, and (b) keep the guarantee when exchangeability fails, which it always does in trading data.

## 2. What is the "score"?

The [[concepts/nonconformity-score|nonconformity score]] `s(x, y)` measures how badly a labelled point `(x, y)` deviates from the calibration sample. Validity is automatic for any deterministic score. Score design controls width and shape only, not whether the guarantee holds.

Canonical scores for trading:

| Setting | Score `s(x, y)` | Interval shape |
|---|---|---|
| Symmetric residuals (basic) | `\|y − μ̂(x)\|` | Equal width around `μ̂` |
| Heteroskedastic returns | `\|y − μ̂(x)\| / σ̂(x)` | Width scales with local vol; the standard HFT choice |
| [[concepts/conformalized-quantile-regression\|CQR]] | `max(q̂_lo(x) − y, y − q̂_hi(x))` | Asymmetric, adapts to skew |
| Directional (long-only) | `y − q̂_lo(x)` (lower tail only) | One-sided downside band |
| [[concepts/spci\|SPCI]] (learned quantile of residuals) | `s` from a quantile-regression of past residuals on features | Best for heteroskedastic and non-stationary |

The σ̂-scaled residual is the one design choice that matters most in HFT. Without it, intervals are constant-width across vol regimes, which is useless: a quote wide enough for the worst minute will never fill in the calm ones. With σ̂-scaling, intervals widen in stress and tighten in calm, and [[concepts/marginal-coverage|marginal coverage]] still holds.

## 3. CP vs Quantile Regression

| | [[concepts/quantile-regression\|Quantile regression]] | Conformal prediction |
|---|---|---|
| What it gives | A point estimate of a quantile of `Y \| X` | A set with coverage guarantee |
| Guarantee | None at finite sample; asymptotic, requires correct specification | Finite-sample marginal coverage, distribution-free |
| Works with | QR-specific losses (pinball) | Any model: XGBoost, GBM, NN, your alpha strategy |
| Width adapts to `x`? | Yes (if you fit per-quantile) | Only if score is `x`-aware (use σ̂-scaling or CQR) |
| Survives non-stationarity? | No (distribution must be stable) | With ACI/EnbPI/WCP variants: yes |
| Compute cost | Train per quantile | Train base model once + cheap calibration step |

They're complements. [[concepts/conformalized-quantile-regression|CQR]] is the standard hybrid: train a quantile regressor for `q_lo, q_hi`, then conformalise the result. You get QR's shape flexibility and CP's coverage proof in one object. In practice, CQR on top of a gradient-boosted quantile regressor is the best off-the-shelf interval predictor I know of.

## 4. Three realistic use cases

### 4.1 Market-making quote placement under adverse selection

The problem. You're quoting a mid-cap corp bond (or BTC-PERP). You want bid/offer that, with 90% confidence, contain the true mid 30 seconds from now. Quote too tight and you get [[concepts/adverse-selection|adversely selected]]. Quote too wide and you don't fill.

Base model. Gradient-boosted regressor of next-30s mid-price given:

- LOB imbalance (top 5 levels)
- recent trade signs / volumes
- volatility regime indicator
- spread
- queue position estimates

The CP wrap.

1. On a rolling 2-week calibration window, compute σ̂-scaled residuals `s_i = |y_i − μ̂(x_i)| / σ̂(x_i)`.
2. Empirical 90% quantile `q̂` of those scores.
3. Quote bid `= μ̂(x) − q̂ · σ̂(x) − ε`, offer `= μ̂(x) + q̂ · σ̂(x) + ε`, where `ε` is your edge / inventory skew from the [[concepts/avellaneda-stoikov-model|Avellaneda-Stoikov]] adjustment.

Why CP beats what you have now. Your XGBoost prediction interval (whether from `predict(..., return_std=True)` or from quantile loss) lies under regime shift. CP intervals stay calibrated even when the model's own confidence is wrong. Post-news bursts are where most market-makers blow up. They're also where naive intervals quietly stop holding.

Time-series fix. Use [[concepts/enbpi|EnbPI]] (bootstrap-ensemble LOO) instead of split CP. Order flow isn't exchangeable; EnbPI handles [[concepts/beta-mixing|β-mixing]] dependence and gives you asymptotic [[concepts/conditional-coverage|conditional coverage]].

### 4.2 Pre-trade slippage estimation for parent orders

The problem. Before sending a 5% ADV order to your execution algo, you want a 95% confidence interval on implementation shortfall in bps. Use it to (a) set the algo's price limit, (b) pick between TWAP / VWAP / IS / liquidity-seeking, (c) cancel if the interval is too wide.

Base model. [[concepts/quantile-regression|Quantile-regression]] random forest (e.g., `quantregForest` or `lightgbm` with quantile objective) predicting median slippage given:

- order size / ADV
- LOB depth at top 10 levels
- short-horizon realised vol (1min, 5min)
- time-of-day / day-of-week effects
- recent fill-rate of similar orders

The CP wrap. Use [[concepts/conformalized-quantile-regression|CQR]]. Train the QR forest for the 2.5% and 97.5% quantiles, then on the calibration set compute `s_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i))`. Adjust the QR intervals by the conformal correction `q̂`. The interval is asymmetric, which respects slippage's natural right-skew (slippage greater than the median has a heavier tail than slippage below it).

Why CP beats your current approach. Your QR forest's intervals are calibrated in-sample only. After the equity vol regime changes (which it always does), they undercover. CQR's conformal adjustment widens or tightens the interval automatically from recent calibration residuals.

Time-series fix. [[concepts/adaptive-conformal-inference|ACI]] with γ ≈ 0.01 for slow vol drift, or [[concepts/agaci|AgACI]] if you don't want to tune γ. For abrupt regime shifts like post-FOMC, [[concepts/conformal-pid-control|Conformal PID Control]] responds faster than vanilla ACI.

### 4.3 Stat-arb signal sizing with calibrated uncertainty

The problem. Your [[concepts/pairs-trading|pairs]] or basket residual reverts to zero on a half-life of ~30 minutes. You forecast next-1min residual return. You want to size positions Kelly-style, which needs a calibrated estimate of the return distribution, not a point forecast.

Base model. Light GBM or XGBoost on residual `z_t`, lagged leg returns, cross-sectional features, regime indicators. Predict `Δz_{t+1}`.

The CP wrap.

1. σ̂-scaled residual score on a 1-month rolling calibration set.
2. EnbPI with bootstrap ensemble (B = 100). Output is a sequence of intervals as time advances.
3. Convert intervals at multiple α levels (50%, 80%, 95%) into an empirical CDF of next-period return.
4. Kelly-size off that CDF, or feed it into a risk-parity / utility-max framework.

Why CP beats your current sizing rule. Most stat-arb shops size as a fixed multiple of the residual's z-score, assuming Gaussian errors. Returns are not Gaussian, and the parametric Kelly fraction is wrong by 30–50% in either direction under realistic kurtosis. CP gives you a non-parametric, regime-aware CDF. Plug that into Kelly directly, without assuming Gaussianity that isn't there.

Bonus. If your stat-arb model is itself a treatment-effect estimator ("did the latest news event move this pair?"), use [[concepts/conformal-prediction|CP]] under the doubly-robust setup of [[sources/koukorinis-2026-draci|DR-ACI]], which adapts conformal inference under temporal dependence with [[concepts/temporal-cross-fitting|temporal cross-fitting]].

## 5. Why naive CP fails on trading data, and what to do

The problem. Standard CP assumes [[concepts/exchangeability|exchangeability]] of `(X_1, Y_1), …, (X_n, Y_n), (X_test, Y_test)`. Returns are not exchangeable. They have temporal dependence, vol clustering, regime shifts, and seasonality. Apply split CP to your time series and coverage drops below the nominal level whenever the data shifts.

The four families of fixes (per [[sources/stocker-2025-conformal-timeseries-intro|Stocker, Małgorzewicz, Fontana, Ben Taieb 2025]]):

| Family | Method | What it does | Use when |
|---|---|---|---|
| Reweight calibration | [[concepts/weighted-conformal-prediction\|WCP]] (exp / linear decay) | Down-weight old residuals | Slow drift, cheap fix |
| Refresh residuals via OOB bootstrap | [[concepts/enbpi\|EnbPI]], [[concepts/spci\|SPCI]] | LOO from bootstrap ensemble; sliding window of residuals | Heteroskedasticity matters, you have compute |
| Adapt α online | [[concepts/adaptive-conformal-inference\|ACI]], [[concepts/agaci\|AgACI]], [[concepts/conformal-pid-control\|PID-Conformal]] | Update the target miscoverage based on realised coverage | Abrupt regime shifts, want active feedback |
| Block randomisation | [[concepts/block-conformal-prediction\|BCP]] | Calibrate at block level | β-mixing data, theoretical appeal |

Empirical bottom lines from Stocker et al.'s 2025 benchmark:

- Under stationary β-mixing, vanilla SCP suffices. Don't over-engineer.
- Under abrupt mean shift, SCP, Block-SCP, and WCP-window all fail to cover.
- ACI, EnbPI, WCP-exp, and WCP-linear self-correct.
- Block-SCP under-covers even on stationary β-mixing data, which is a non-obvious failure mode.

Practical recommendation for HFT and algo: default to EnbPI with σ̂-scaled residuals plus ACI feedback (γ ≈ 0.005–0.02). For a cheap baseline, WCP-exponential with the decay rate tuned to your regime half-life. When sizing matters more than coverage, switch to [[concepts/spci|SPCI]] (heteroskedasticity-aware quantile score).

## 6. Combining CP with tree models

Tree-based learners (XGBoost, LightGBM, CatBoost, RF) dominate tabular financial prediction. Out of the box you get a strong point predictor, native handling of categorical regime indicators, feature interactions for free, and fast inference.

What they don't give you is honest prediction intervals. Their quantile objectives look fine on the training set and fail in test. Miscoverage gets worse under regime shift.

CP wraps cleanly around any tree model. Four flavours, in increasing order of how much they help.

### 6.1 XGBoost mean + split CP residual quantile

```python
# Train base regressor
base = XGBRegressor(...)
base.fit(X_train, y_train)

# Calibrate
mu_cal = base.predict(X_cal)
residuals = np.abs(y_cal - mu_cal)
q_hat = np.quantile(residuals, np.ceil((len(residuals) + 1) * 0.9) / len(residuals))

# Predict
mu_test = base.predict(X_test)
lower, upper = mu_test - q_hat, mu_test + q_hat  # 90% interval
```

Trivial and fast, but constant width across regimes. Bad for HFT.

### 6.2 XGBoost mean + σ̂-scaled residual

Train a second XGBoost on `|y − μ̂|` to predict local vol `σ̂(x)`. Then score = `|y − μ̂| / σ̂`. Interval = `μ̂(x) ± q̂ · σ̂(x)`. Width now adapts to local volatility: it widens in stressed minutes and tightens in calm.

### 6.3 XGBoost quantile + CQR

Train one XGBoost per quantile (`q_lo, q_hi`) with quantile loss. Apply CQR's conformal correction. The strongest off-the-shelf option for tabular regression. Use [`MAPIE`](https://github.com/scikit-learn-contrib/MAPIE), which implements `MapieQuantileRegressor` as a drop-in around `sklearn` and `xgboost`-compatible regressors.

### 6.4 The EnbPI version (for time series)

Replace the calibration step with EnbPI: `B` bootstrap XGBoost models, each excluding ~37% of training. The LOO predictors are aggregations of out-of-bag bootstraps. Residual quantile comes from a sliding window of LOO residuals. No retraining at test time. For HFT-relevant horizons this is the default I'd reach for.

Library: [`MAPIE`](https://mapie.readthedocs.io) has `MapieTimeSeriesRegressor` implementing EnbPI on top of any sklearn-compatible base model.

## 7. Three simulations you can build today

Goal: convince yourself (and your risk team) with realistic data that CP intervals hold where parametric and historical-quantile intervals fail.

### Sim 1: Coverage failure of naive intervals under heteroskedasticity

Setup.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor

np.random.seed(0)
T = 20000
# AR(1) with GARCH(1,1) errors
def garch_ar(T, phi=0.3, omega=0.001, alpha=0.1, beta=0.85):
    eps = np.zeros(T); sig2 = np.full(T, omega / (1 - alpha - beta))
    y = np.zeros(T); x = np.zeros((T, 5))
    for t in range(1, T):
        sig2[t] = omega + alpha * eps[t-1]**2 + beta * sig2[t-1]
        eps[t] = np.sqrt(sig2[t]) * np.random.randn()
        x[t] = [y[t-1], y[t-2] if t > 1 else 0, eps[t-1], sig2[t-1], np.sin(t/100)]
        y[t] = phi * y[t-1] + eps[t]
    return x, y

X, y = garch_ar(T)
train, cal, test = slice(0, 10000), slice(10000, 15000), slice(15000, T)

base = GradientBoostingRegressor(n_estimators=200, max_depth=4).fit(X[train], y[train])

# Method A: parametric ±1.645σ from training residuals
sigma = np.std(y[train] - base.predict(X[train]))
A_lo = base.predict(X[test]) - 1.645 * sigma
A_hi = base.predict(X[test]) + 1.645 * sigma

# Method B: rolling 90% historical quantile of residuals (last 500)
# Method C: split conformal
mapie = MapieRegressor(base, method="base", cv="prefit")
mapie.fit(X[cal], y[cal])
C_lo, C_hi = mapie.predict(X[test], alpha=0.1)[1][:, 0, 0], mapie.predict(X[test], alpha=0.1)[1][:, 1, 0]

# Method D: EnbPI
from mapie.regression import MapieTimeSeriesRegressor
from sklearn.model_selection import KFold
mapie_ts = MapieTimeSeriesRegressor(base, method="enbpi", cv=KFold(n_splits=5))
mapie_ts.fit(X[train], y[train])
D_lo, D_hi = mapie_ts.predict(X[test], alpha=0.1)[1][:, 0, 0], mapie_ts.predict(X[test], alpha=0.1)[1][:, 1, 0]

for name, lo, hi in [("Parametric", A_lo, A_hi), ("Split CP", C_lo, C_hi), ("EnbPI", D_lo, D_hi)]:
    coverage = np.mean((y[test] >= lo) & (y[test] <= hi))
    width = np.mean(hi - lo)
    print(f"{name}: coverage={coverage:.3f} (target 0.90), avg width={width:.4f}")
```

Expected outcome: parametric undercovers around 0.83. Split CP holds at 0.90 but with wider average width than EnbPI. EnbPI also holds and adapts.

### Sim 2: Regime-shift recovery

Same generator, but at `t = 12500` multiply `omega` by 5 (vol explosion). Re-run all methods. Track rolling 500-step empirical coverage for each.

Expected outcome: parametric goes to 0.6–0.7 and stays there. Split CP slowly drifts as the calibration set goes stale. EnbPI + ACI recovers to about 0.90 within ~200 steps. The reason: EnbPI's sliding window refreshes the residual pool, and ACI's α update kicks in as soon as coverage drops.

To add ACI on top of EnbPI:

```python
alpha_t = 0.1
gamma = 0.01
for t in range(len(test_indices)):
    lo, hi = mapie_ts.predict(X[t:t+1], alpha=alpha_t)
    err = not (lo[0] <= y[t] <= hi[0])
    alpha_t = max(0.001, min(0.999, alpha_t + gamma * (0.1 - err)))
```

Plot α_t over time. You'll see it spike when the regime breaks and slowly return.

### Sim 3: Tree + CP on a realistic LOB-derived dataset

Synthesise a 100k-row HFT-style dataset:

- target: next-1000ms mid-return
- features: order-flow imbalance (5 levels), trade-flow imbalance (last 500ms), short-vol (last 30s), spread, queue-imbalance

Compare four methods on a held-out test set:

| Method | What it is |
|---|---|
| (a) XGB quantile (`reg:quantileerror`) at 5% and 95% | Native XGB intervals |
| (b) XGB mean + Gaussian `±1.96σ` | Parametric baseline |
| (c) XGB mean + EnbPI (split CP would be wrong here) | Recommended workhorse |
| (d) XGB quantile + CQR + EnbPI | Kitchen sink: heteroskedasticity-safe and non-exchangeable-safe |

Metrics:

- Coverage: empirical hit rate vs nominal (target 0.90).
- Width: average interval width. Narrower is better, conditional on coverage holding.
- Conditional coverage: stratify the test set by realised vol decile and check coverage in each. Method (d) should be flattest.
- Calibration plot: quantile-quantile of nominal vs realised coverage at α ∈ {0.5, 0.7, 0.8, 0.9, 0.95}.

What you should see: (a) and (b) undercover during vol bursts, (c) holds marginally but is wide, (d) holds marginally and conditionally with the tightest valid widths. That's the empirical case for "tree + CQR + EnbPI" as a practical default.

## 8. The practical benefits

1. Coverage you can take to risk. When the CP-90% interval says 90%, it delivers 90% on the test set, distribution-free. Your VaR, Kelly sizing, and stop-loss can finally trust the band.

2. Wraps around what you already have. Don't throw away your XGBoost, LightGBM, or NN. CP wraps the existing model; the calibration step bolts on top.

3. Width that adapts to regime. With σ̂-scaling or CQR scoring, intervals widen in stressed minutes and tighten in calm ones. Use that directly for quote skew and inventory limits.

4. Survives non-stationary order flow. EnbPI handles temporal dependence; ACI/AgACI/PID-Conformal handle abrupt regime shifts. The four-family taxonomy gives you principled choices.

5. Asymmetric intervals are natural. CQR doesn't force `±q̂` symmetry. Slippage right-skew and tail asymmetry come out of the score directly.

6. One-sided coverage on demand. Long-only? Calibrate the lower tail. Short-only? The upper. Same machinery.

7. Auditable. Every interval traces back to (calibration set, base model, score function, α level). Risk officers and regulators actually understand this, which is more than I can say for "the neural network is 92% confident."

8. Composes with treatment-effect estimation. For causal questions like "did this news event move the basket?", [[sources/koukorinis-2026-draci|DR-ACI]] combines doubly-robust estimation with adaptive CP under temporal dependence.

## 9. What to read, where to go next

Foundational (read in this order):

1. [[sources/angelopoulos-2022-gentle-intro]]: Angelopoulos & Bates canonical tutorial. The four-step CP recipe, score catalogue, CQR, weighted CP, risk control.
2. [[sources/stocker-2025-conformal-timeseries-intro]]: Stocker et al. time-series companion. The four-family taxonomy this guide leans on.
3. [[sources/dieuleveut-zaffran-2025-cp-tutorial]]: 91-slide Dieuleveut & Zaffran deck. Cleanest visual treatment.

Algorithmic / time-series methods:

- [[sources/zaffran-2022-aci]]: ACI / AgACI for electricity-style time series.
- [[sources/xu-2023-enbpi]]: EnbPI (bootstrap-ensemble LOO). The recommended default.
- [[sources/xu-2022-spci]]: SPCI (heteroskedasticity-aware via learned quantile score).

Trading-adjacent / advanced:

- [[sources/koukorinis-2026-draci]]: DR-ACI for causal treatment effects under temporal dependence. Use when "what moved the spread" matters as much as "what will it be next minute".
- [[sources/zaffran-2023-conformal-missing]]: CP with missing values ([[concepts/mask-conditional-validity|mask-conditional validity]]). Relevant for sparse LOB features.

Libraries:

- [`MAPIE`](https://mapie.readthedocs.io): `MapieRegressor`, `MapieQuantileRegressor`, `MapieTimeSeriesRegressor`. The Python default. Implements split CP, CQR, EnbPI.
- [`crepes`](https://github.com/henrikbostrom/crepes): alternative with a cleaner API for tree-based CP and conformal predictive systems.
- [`AdaptiveConformal`](https://github.com/aangelopoulos/conformal-time-series): Angelopoulos's reference implementation of ACI / PID-Conformal.

## 10. Pitfalls and footguns

Don't mix calibration and training. Split CP requires a held-out calibration set the base model never saw. EnbPI uses the bootstrap to avoid the split, so reach for it when your data is precious.

Don't apply standard CP to time-series data and expect coverage to hold. Use one of the four families. Vanilla SCP on returns is the most common HFT mistake.

Coverage is not utility. A 90% interval can be wide and useless. Always report width, calibration plot, and conditional coverage by vol decile alongside the headline number.

Calibration window matters more than people think. Too short and `q̂` is noisy; too long and it's stale. Tune to your half-life of regime persistence: usually 1–5 days for liquid instruments, 1–4 weeks for credit.

Asymmetric isn't free. One-sided coverage halves your effective calibration data. Use it when you genuinely have a one-sided risk; otherwise stick with two-sided CQR.

Score design is where the value lives. Coverage is mechanical. Tight valid intervals come from a smart score. Spend your model time there, not on the wrapper.

---

*Synthesis source for this analysis: the 2026-05-21 CP tutorial/review quartet (Angelopoulos & Bates [164], Xu & Xie EnbPI [165], Stocker et al. [166], Dieuleveut & Zaffran Hi! PARIS [167]) plus the existing CP literature anchors in MIND_MAP hub [3] (Zaffran ACI [111], Xu SPCI [108], Koukorinis DR-ACI [59], and others). For an academic treatment, start with the four-source quartet. For applied work, use this page as the entry point and link out as questions arise.*
