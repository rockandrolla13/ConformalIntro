# Figures Guide: Conformal Prediction for HFT Traders

This guide groups the 13 figures in `figures/` into **four practitioner-oriented batches**, each answering one question a trading-floor or research-team audience will ask. Use it to choose figures for slides, papers, internal explainers, or debugging sessions — not all 13 belong in every venue.

For each group: **what it shows**, **the question it answers**, **who the audience is**, and **how to use it in practice**.

---

## Group 1 — Motivate: "Why do naive intervals fail?"

**The three figures that justify the whole project. Always show at least one before introducing conformal prediction.**

| File | Caption (one-liner) |
|---|---|
| `sim1_series_and_vol.png` | A GARCH-AR return path with the volatility process overlaid; bands of changing scale are visible by eye. |
| `sim2_regime_break_vol.png` | Same idea but with a structural break — volatility doubles at the midpoint, mimicking a regime shift like Mar 2020. |
| `sim3_synthetic_series.png` | An LOB-derived feature/return panel: signed order-flow imbalance and forward returns, with bursty clustering typical of HFT data. |

**Question answered.** *Does this look stationary and homoskedastic to you?* No — and therefore the textbook "fit a model, take ±1.96·σ" approach is going to mis-cover wildly. These three plots make the failure mode visceral *before* introducing the math.

**Audience.** Practitioners new to CP. Risk managers who think VaR + a normal assumption is enough. Quant researchers being onboarded.

**How to use in practice.**
- **Slide deck (intro):** Pick **one** figure that matches your audience's asset class — equity vol traders see themselves in `sim1`, macro/rates teams in `sim2`, market-makers in `sim3`. Don't show all three.
- **Code review / model design discussion:** Pin `sim2_regime_break_vol.png` to the discussion as a "what we need our intervals to survive."
- **Skip** in COPA-style theory talks — the audience already accepts heteroskedasticity is the problem.

---

## Group 2 — Diagnose: "Where exactly does naive CP fall apart?"

**Close-up failure-mode evidence. Use after you've sketched CP's headline `1−α` guarantee and someone asks 'but in practice…?'**

| File | Caption (one-liner) |
|---|---|
| `sim1_intervals_zoom.png` | Zoomed window of split-CP intervals on the heteroskedastic series — constant-width bands miss systematically inside high-vol regions. |
| `sim1_conditional_coverage.png` | Coverage stratified by volatility quintile: marginal coverage is at 90% but high-vol bin is at ~70%. |
| `sim3_conditional_coverage.png` | Same diagnostic on the LOB problem, stratified by absolute order-flow imbalance — coverage collapses for the most informative buckets. |

**Question answered.** *Does my 90% interval actually deliver 90% coverage when it matters — i.e., conditional on the regime being adverse?* Almost always no, for naive split-CP. The marginal guarantee averages over easy and hard regions, and your trading PnL lives in the hard region.

**Audience.** Risk and oversight functions. Quant researchers who got the marginal-coverage talk last week and now want to see the catch.

**How to use in practice.**
- **Investment committee / model risk review:** `sim1_conditional_coverage.png` is the single most important figure. It refutes the bogus "we have 90% coverage so we're fine" claim.
- **Post-mortem on a coverage breach:** Use `sim1_intervals_zoom.png` to show a specific window where the bands were obviously too narrow.
- **Paper writing:** Pair `sim1_conditional_coverage` with `sim3_conditional_coverage` — same failure mode in two different data-generating processes shows generality.
- **Selling CQR/EnbPI internally:** Show this group right *before* introducing the fix. The contrast does the persuasion work.

---

## Group 3 — Adapt: "How do online methods recover after a regime change?"

**This is the most important group for time-series practitioners. ACI / quantile-tracker is the workhorse that makes CP usable for live trading.**

| File | Caption (one-liner) |
|---|---|
| `sim2_aci_alpha_trajectory.png` | The α parameter being driven by the online update rule — visible jump up at the regime break, then convergence. |
| `sim2_rolling_coverage.png` | 250-day rolling coverage: split-CP drops to ~75% after the break and stays there; ACI returns to 90% within ~50 obs. |
| `sim2_width_over_time.png` | The cost: ACI's width widens during the transition and stays modestly higher post-break — the price of adaptivity. |

**Question answered.** *When the world changes — vol explodes, microstructure shifts, a regime breaks — how fast does my interval system notice and what does it cost in width?* Together these three show the response curve.

**Audience.** Production-quant teams. Anyone deploying CP intervals on a daily-or-faster cadence. Reviewers who want evidence the method survives non-stationarity.

**How to use in practice.**
- **Production playbook:** All three figures belong in your runbook. They establish the recovery time SLA (~50 obs in this sim — your real number depends on the learning rate and break magnitude).
- **Parameter selection:** When tuning the ACI learning rate γ, regenerate `sim2_rolling_coverage.png` and `sim2_width_over_time.png` for each candidate. The trade-off you're picking is on this width-vs-coverage curve.
- **Slide deck (technical talk):** Show all three side-by-side as a triptych — α moves, coverage recovers, width pays. The story is complete in one slide.
- **Debugging a misbehaving live system:** If your rolling-coverage chart looks like the *split-CP* line in `sim2_rolling_coverage` rather than the *ACI* line, you forgot to turn on adaptation.

---

## Group 4 — Validate: "Is the method ready for deployment? Calibrate, compare, commit."

**Pre-deployment checks. These figures answer 'should I ship this?'**

| File | Caption (one-liner) |
|---|---|
| `sim1_coverage_width.png` | Coverage vs. width Pareto frontier across split-CP / CQR / σ̂-scaled — informs the efficiency choice. |
| `sim3_marginal_coverage_width.png` | Same comparison on the LOB problem; CQR + tree mean dominates. |
| `sim3_calibration_plot.png` | Forecast-quantile vs. realized-quantile reliability diagram on held-out data. |
| `figures/tutorial/tutorial_02_residual_quantile.png` | The four-panel split-CP recipe: train → residual histogram → empirical quantile → calibrated band. The "how the sausage is made" illustration. |

**Question answered.** *Among the CP variants, which is the most efficient (narrowest interval at target coverage) for my data — and is its calibration honest?* `coverage_width` plots give the comparison; the calibration plot is the honesty check.

**Audience.** Yourself, the night before promotion to production. Reviewers (paper or model-risk). Quant teams choosing between split-CP, CQR, conditional CQR, and EnbPI.

**How to use in practice.**
- **Pre-deployment gate:** A model doesn't ship unless its `sim*_marginal_coverage_width.png` plot shows it on the Pareto frontier *and* `sim3_calibration_plot.png` shows reliability points hugging the 45° line.
- **Method-selection memo:** `sim1_coverage_width.png` paired with `sim3_marginal_coverage_width.png` — same comparison across two DGPs — settles arguments about which CP variant to default to.
- **Onboarding new researchers:** `tutorial_02_residual_quantile.png` (the four-panel recipe) is the *single* figure that explains how split-CP works. Use it as the very first slide of any internal training.
- **Paper appendix:** All four go in the appendix as part of the empirical-section robustness checks.

---

## Quick-reference matrix: figure → use-case

| Use case | Figures to show |
|---|---|
| 5-min elevator pitch ("why CP?") | Group 1 (pick one) + `sim1_conditional_coverage.png` |
| Internal tutorial (90 min) | `tutorial_02_residual_quantile.png` → Group 2 → Group 3 |
| Risk-committee defense of a deployed system | Group 2 (`sim1_conditional_coverage` only) + Group 3 (all) |
| Paper main body | One per group (4 total) |
| Paper appendix / robustness | Remaining 9 figures |
| Production runbook | Group 3 (all) + `sim3_calibration_plot.png` |
| Choosing ACI vs. split-CP for production | Group 3 (all) |
| Choosing CQR vs. split-CP for production | `sim1_coverage_width.png`, `sim3_marginal_coverage_width.png` |
| Debugging a coverage breach | `sim1_intervals_zoom.png` + the regime/vol figure that matches your asset |

---

## How the figures were produced

All figures are deterministic. To regenerate any of them:

```bash
cd notebooks/
jupyter execute 01_sim1_heteroskedastic_coverage.ipynb  # produces sim1_*.png
jupyter execute 02_sim2_regime_shift_aci.ipynb          # produces sim2_*.png
jupyter execute 03_sim3_lob_benchmark.ipynb             # produces sim3_*.png
python tutorial_figures.py                              # produces figures/tutorial/*
```

`utils.py` contains the shared DGPs (`garch_ar`, regime-break process, LOB-style simulator) and the split-CP / EnbPI implementations. Seeds are pinned in each notebook.

---

## What's deliberately *not* here

Three figures you might expect but won't find — and why:

1. **A backtest equity curve.** Coverage is a calibration property; PnL is a strategy property. Confusing them is a frequent footgun. If you need a PnL chart, build a separate strategy notebook.
2. **A "comparison to BlackScholes/parametric VaR" plot.** That's a different conversation. Compare against parametric methods only when the audience has bought into the CP framework first.
3. **A Sharpe-ratio table.** Same reason as #1 — and Sharpe is the *last* thing to look at when assessing a probabilistic forecaster.
