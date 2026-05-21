"""
Shared utilities for the Conformal-for-HFT notebooks.

Synthetic data generators, a compact EnbPI implementation, and a `save_fig`
helper that mirrors every figure to both the local figures/ folder and the
wiki assets/ folder. Reference implementations of ACI / quantile-tracker /
PID are imported from the upstream conformal-time-series repo when available.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths --------------------------------------------------------------------
PROJECT_ROOT = Path("/media/ak/10E1026C4FA6006E/GitRepos/ConformalIntro")
FIGURES_DIR = PROJECT_ROOT / "figures"
WIKI_ASSETS_DIR = Path(
    "/media/ak/10E1026C4FA6006E/GitRepos/LLMWikiGeneration/wiki/wiki/assets/conformal-hft"
)
UPSTREAM_REPO = Path("/media/ak/10E1026C4FA6006E/GitRepos/conformal-time-series")

# Make upstream `core` importable as `cts_core` if we ever want it.
if str(UPSTREAM_REPO) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_REPO))


# Plot style ---------------------------------------------------------------
def setup_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save fig to both ConformalIntro/figures/ and wiki assets/. Returns local path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    local = FIGURES_DIR / name
    wiki = WIKI_ASSETS_DIR / name
    fig.savefig(local)
    shutil.copyfile(local, wiki)
    return local


# Synthetic data generators ------------------------------------------------
@dataclass
class GarchArSeries:
    X: np.ndarray
    y: np.ndarray
    sig2: np.ndarray
    regime_break: int | None = None


def garch_ar(
    T: int = 20_000,
    phi: float = 0.3,
    omega: float = 1e-3,
    alpha: float = 0.1,
    beta: float = 0.85,
    regime_break: int | None = None,
    regime_omega_mult: float = 5.0,
    seed: int = 0,
) -> GarchArSeries:
    """
    AR(1) mean dynamics with GARCH(1,1) shocks.
    Optional regime break at `regime_break` that multiplies omega by `regime_omega_mult`.
    Returns features X (5 lag/vol/seasonal columns), target y, latent sig2.
    """
    rng = np.random.default_rng(seed)
    eps = np.zeros(T)
    sig2 = np.full(T, omega / max(1 - alpha - beta, 1e-6))
    y = np.zeros(T)
    X = np.zeros((T, 5))
    omega_t = omega
    for t in range(1, T):
        if regime_break is not None and t == regime_break:
            omega_t = omega * regime_omega_mult
        sig2[t] = omega_t + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * rng.standard_normal()
        X[t] = [
            y[t - 1],
            y[t - 2] if t > 1 else 0.0,
            eps[t - 1],
            sig2[t - 1],
            np.sin(t / 100.0),
        ]
        y[t] = phi * y[t - 1] + eps[t]
    return GarchArSeries(X=X, y=y, sig2=sig2, regime_break=regime_break)


@dataclass
class LobSeries:
    X: pd.DataFrame
    y: np.ndarray
    vol: np.ndarray  # latent vol for stratification


def make_lob_dataset(
    T: int = 100_000,
    seed: int = 7,
) -> LobSeries:
    """
    Synthetic HFT-style features and a next-step mid-return target.

    Features:
        ofi          order-flow imbalance (5-level aggregate, in [-1, 1])
        tfi          trade-flow imbalance (last 500ms proxy)
        rv_30s       short realised vol (rolling)
        spread       bid-ask spread (positive)
        queue_imb    queue imbalance at top of book ([-1, 1])

    The target is heteroskedastic with a sign loosely driven by OFI/TFI, vol clusters,
    and intermittent regime shifts so the data isn't exchangeable.
    """
    rng = np.random.default_rng(seed)
    # Latent vol process with mild regime shifts at deterministic anchors.
    log_vol = np.zeros(T)
    log_vol[0] = -3.0
    shift_idx = {T // 4, T // 2, 3 * T // 4}
    for t in range(1, T):
        shock = rng.normal(0, 0.05)
        if t in shift_idx:
            shock += rng.choice([-0.6, 0.6])
        log_vol[t] = 0.995 * log_vol[t - 1] + shock
    vol = np.exp(log_vol)

    ofi = np.clip(rng.normal(0, 0.4, T), -1, 1)
    tfi = np.clip(0.6 * ofi + rng.normal(0, 0.4, T), -1, 1)
    spread = np.exp(rng.normal(-4 + 0.2 * np.abs(log_vol), 0.15))
    queue_imb = np.clip(rng.normal(0, 0.5, T), -1, 1)
    rv_30s = pd.Series(vol).rolling(30, min_periods=1).mean().to_numpy()

    signal = 0.45 * ofi + 0.25 * tfi - 0.15 * queue_imb
    eps = rng.standard_normal(T) * vol
    y = signal * vol + eps  # return scales with vol

    X = pd.DataFrame(
        {
            "ofi": ofi,
            "tfi": tfi,
            "rv_30s": rv_30s,
            "spread": spread,
            "queue_imb": queue_imb,
        }
    )
    return LobSeries(X=X, y=y, vol=vol)


# Conformal pieces ---------------------------------------------------------
def split_cp_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1-alpha) quantile of |residuals| for split CP."""
    n = len(residuals)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(np.abs(residuals), level, method="higher"))


def enbpi_predict(
    base_cls,
    base_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    B: int = 20,
    alpha: float = 0.1,
    seed: int = 0,
    sliding_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compact EnbPI (Xu & Xie 2023) on a single train/test split.

    Returns (mu_test, lower, upper) where lower/upper use a sliding-window
    quantile of out-of-bag LOO residuals computed on the training set.
    For tests, the residual quantile is held fixed at the last training
    residuals; for fully-online use, recompute after each test step.

    Args:
        base_cls: estimator class with .fit, .predict
        base_kwargs: kwargs for base_cls(**base_kwargs)
        sliding_window: if not None, use only the last `sliding_window` LOO residuals.
    """
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    # Bootstrap indices
    boot_idx = [rng.integers(0, n_train, n_train) for _ in range(B)]
    # Fit ensemble
    models = []
    in_bag = np.zeros((B, n_train), dtype=bool)
    for b in range(B):
        in_bag[b, boot_idx[b]] = True
        model = base_cls(**base_kwargs)
        model.fit(X_train[boot_idx[b]], y_train[boot_idx[b]])
        models.append(model)

    # LOO residuals: average of bootstraps that did NOT see i
    train_preds = np.stack([m.predict(X_train) for m in models], axis=0)  # (B, n)
    loo_mu = np.full(n_train, np.nan)
    for i in range(n_train):
        oob = ~in_bag[:, i]
        if oob.any():
            loo_mu[i] = train_preds[oob, i].mean()
    valid = ~np.isnan(loo_mu)
    residuals = np.abs(y_train[valid] - loo_mu[valid])
    if sliding_window is not None and len(residuals) > sliding_window:
        residuals = residuals[-sliding_window:]
    q_hat = np.quantile(
        residuals,
        np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals),
        method="higher",
    )

    # Test prediction: average across ensemble
    test_preds = np.stack([m.predict(X_test) for m in models], axis=0).mean(axis=0)
    lower = test_preds - q_hat
    upper = test_preds + q_hat
    return test_preds, lower, upper


def enbpi_sliding_predict(
    base_cls,
    base_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    B: int = 20,
    alpha: float = 0.1,
    window: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EnbPI with a true online sliding-window of residuals: each step appends the
    most recent absolute residual to the residual pool and recomputes q_hat.

    Returns (mu_test, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    boot_idx = [rng.integers(0, n_train, n_train) for _ in range(B)]
    models = []
    in_bag = np.zeros((B, n_train), dtype=bool)
    for b in range(B):
        in_bag[b, boot_idx[b]] = True
        model = base_cls(**base_kwargs)
        model.fit(X_train[boot_idx[b]], y_train[boot_idx[b]])
        models.append(model)

    train_preds = np.stack([m.predict(X_train) for m in models], axis=0)
    loo_mu = np.full(n_train, np.nan)
    for i in range(n_train):
        oob = ~in_bag[:, i]
        if oob.any():
            loo_mu[i] = train_preds[oob, i].mean()
    valid = ~np.isnan(loo_mu)
    pool = list(np.abs(y_train[valid] - loo_mu[valid]))[-window:]

    test_preds = np.stack([m.predict(X_test) for m in models], axis=0).mean(axis=0)
    lower = np.zeros_like(test_preds)
    upper = np.zeros_like(test_preds)
    for t in range(len(X_test)):
        arr = np.asarray(pool[-window:])
        q_hat = np.quantile(
            arr,
            np.ceil((len(arr) + 1) * (1 - alpha)) / len(arr),
            method="higher",
        )
        lower[t] = test_preds[t] - q_hat
        upper[t] = test_preds[t] + q_hat
        # Append the new residual after observation
        pool.append(abs(y_test[t] - test_preds[t]))
    return test_preds, lower, upper


def aci_update(
    residuals_history: np.ndarray,
    y_obs: np.ndarray,
    mu: np.ndarray,
    alpha_target: float = 0.1,
    gamma: float = 0.01,
    window: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gibbs–Candès Adaptive Conformal Inference using |y - mu| residuals.
    Returns (lower, upper, alpha_t trajectory). Online: at each step, builds
    the interval from a sliding-window quantile at alpha_t, then updates
    alpha_t based on whether the realised obs was covered.
    """
    T = len(mu)
    alpha_t = alpha_target
    alphas = np.zeros(T)
    lower = np.zeros(T)
    upper = np.zeros(T)
    pool = list(residuals_history)
    for t in range(T):
        arr = np.asarray(pool[-window:]) if len(pool) >= 1 else np.array([1.0])
        a_eff = float(np.clip(alpha_t, 1e-3, 0.999))
        q_hat = np.quantile(arr, 1 - a_eff, method="higher")
        lower[t] = mu[t] - q_hat
        upper[t] = mu[t] + q_hat
        covered = (lower[t] <= y_obs[t]) and (y_obs[t] <= upper[t])
        # ACI update: alpha_{t+1} = alpha_t + gamma*(alpha_target - err)
        err = 0.0 if covered else 1.0
        alpha_t = alpha_t + gamma * (alpha_target - err)
        alphas[t] = a_eff
        pool.append(abs(y_obs[t] - mu[t]))
    return lower, upper, alphas


# Metric helpers -----------------------------------------------------------
def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def rolling_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, win: int = 500) -> np.ndarray:
    hit = ((y >= lo) & (y <= hi)).astype(float)
    return pd.Series(hit).rolling(win, min_periods=max(50, win // 5)).mean().to_numpy()
