"""Noise-robust realized variance estimators.

Implements the standard microstructure-noise-robust alternatives to
simple RV = Σ r_i²:

    SimpleRV          Andersen-Bollerslev (1998)         — baseline
    SubsampledRV      Zhang-Mykland-Aït-Sahalia (2005)   — bias reduction
    TSRV              Zhang-Mykland-Aït-Sahalia (2005)   — Two-Scale RV
    RealizedKernel    Barndorff-Nielsen et al. (2008)    — kernel-weighted
    BipowerVariation  Barndorff-Nielsen-Shephard (2004)  — jump-robust
    SignaturePlot     Bandi-Russell (2008)               — diagnostic

Each estimator takes a 1-D array of intra-day log-returns (one day).
For multi-day computation, group by date externally and call per-day.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ======================================================================
# Constants
# ======================================================================
MU1 = float(np.sqrt(2.0 / np.pi))  # E|Z| for Z ~ N(0,1)


# ======================================================================
# 1. Simple RV (baseline)
# ======================================================================
def simple_rv(r: np.ndarray) -> float:
    """Realized variance = Σ r_i². Andersen-Bollerslev (1998)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.sum(r * r))


# ======================================================================
# 2. Sub-sampled RV (averaged over K offset grids)
# ======================================================================
def subsampled_rv(r: np.ndarray, K: int = 5) -> float:
    """Sub-sampled RV: average of K offset RVs at K-period aggregated returns.

    Given M one-step returns and aggregation rate K, builds K offset
    grids of K-period returns by summing consecutive blocks of K
    returns at each phase offset 0..K-1. Computes simple RV on each
    sparse grid and averages. This is the correct slow-scale RV used
    in the TSRV estimator.

    Reference: Zhang-Mykland-Aït-Sahalia (2005).
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    M = len(r)
    if M < 2 * K:
        return float('nan')
    rvs = []
    for offset in range(K):
        sub_returns = r[offset:]
        n_groups = len(sub_returns) // K
        if n_groups < 2:
            continue
        agg = sub_returns[:n_groups * K].reshape(n_groups, K).sum(axis=1)
        rvs.append(float(np.sum(agg * agg)))
    if not rvs:
        return float('nan')
    return float(np.mean(rvs))


# ======================================================================
# 3. Two-Scale Realized Volatility (TSRV)
#    TSRV = RV_slow − (n̄/n) · RV_all
# ======================================================================
def two_scale_rv(r: np.ndarray, K: int = 5) -> float:
    """Two-Scale RV (Zhang-Mykland-Aït-Sahalia 2005).

    Combines a sub-sampled RV (slow scale) with the finest-frequency
    RV (fast scale) and removes the leading-order noise bias:

        TSRV = (1 − n̄/n)⁻¹ · ( RV_slow − (n̄/n) · RV_all )

    where:
        RV_all  = Σ_{i=1}^{n} r_i²                (finest frequency)
        RV_slow = average of K sub-sampled RVs    (sub-sample rate K)
        n̄      = avg # obs per sub-grid = (n − K + 1)/K
        n       = total # of fine-frequency observations
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < K + 5:
        return float('nan')
    rv_all = float(np.sum(r * r))
    rv_slow = subsampled_rv(r, K=K)
    n_bar = (n - K + 1) / K
    ratio = n_bar / n
    if ratio >= 0.999:
        return rv_slow
    return float((rv_slow - ratio * rv_all) / (1.0 - ratio))


# ======================================================================
# 4. Realized Kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard, 2008)
# ======================================================================
def _parzen_kernel(x: np.ndarray) -> np.ndarray:
    """Parzen kernel: standard choice for realized kernel (PSD-preserving)."""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    m1 = x <= 0.5
    m2 = (x > 0.5) & (x <= 1.0)
    out[m1] = 1.0 - 6.0 * x[m1] ** 2 + 6.0 * x[m1] ** 3
    out[m2] = 2.0 * (1.0 - x[m2]) ** 3
    return out


def realized_kernel(
    r: np.ndarray,
    H: Optional[int] = None,
    kernel: str = 'parzen',
) -> float:
    """Realized Kernel = γ₀ + Σ_{h=1}^{H} k((h-1)/H) · (γ_h + γ_{-h}).

    Parameters
    ----------
    r : array of log-returns on the finest sampling grid
    H : kernel bandwidth (rule of thumb: H ≈ c·n^{2/5}, c≈3 for Parzen)
    kernel : 'parzen' (default; PSD-preserving)

    Returns
    -------
    float realized-kernel variance estimate (jump-corrupted; corrected
    for microstructure noise to leading order).

    References
    ----------
    Barndorff-Nielsen, Hansen, Lunde & Shephard (2008),
    "Designing Realised Kernels to Measure Ex-Post Variation of
    Equity Prices in the Presence of Noise."
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return float('nan')
    if H is None:
        # BNHLS optimal-bandwidth rule of thumb
        H = max(2, int(np.floor(3.5134 * n ** (2.0 / 5.0))))
    H = min(H, n - 1)
    # γ_0 = Σ r²
    gamma0 = float(np.sum(r * r))
    # γ_h = Σ_{i=1}^{n-h} r_i · r_{i+h}, for h = 1..H
    rk = gamma0
    if kernel != 'parzen':
        raise ValueError(f"Unknown kernel {kernel!r}")
    for h in range(1, H + 1):
        gamma_h = float(np.sum(r[:n - h] * r[h:]))
        w = float(_parzen_kernel(np.array([(h - 1) / H]))[0])
        rk += 2.0 * w * gamma_h
    return rk


# ======================================================================
# 5. Bipower Variation (already in realized.py; re-export for symmetry)
# ======================================================================
def bipower_variation(r: np.ndarray) -> float:
    """BPV = (π/2) · Σ |r_i| · |r_{i-1}|.  Barndorff-Nielsen & Shephard (2004)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float('nan')
    abs_r = np.abs(r)
    return float((1.0 / MU1 ** 2) * np.sum(abs_r[1:] * abs_r[:-1]))


# ======================================================================
# 6. Signature plot — RV as a function of sampling frequency
# ======================================================================
def signature_plot_one_day(
    log_prices: np.ndarray,
    freqs_in_finest_units: Sequence[int],
) -> pd.DataFrame:
    """For one day of finest-frequency log-prices, compute RV at multiple
    sub-sampled frequencies.

    Parameters
    ----------
    log_prices : array
        Log-prices on the finest grid (e.g., 1-minute closes).
    freqs_in_finest_units : sequence of int
        E.g. (1, 5, 15, 30, 60) means: sample every 1st, 5th, ..., 60th
        observation. For 1-min input, these are minute multiples.

    Returns
    -------
    DataFrame with columns ('freq', 'rv', 'rk', 'tsrv', 'n_bars').
    """
    log_prices = np.asarray(log_prices, dtype=float)
    log_prices = log_prices[np.isfinite(log_prices)]
    rows = []
    for k in freqs_in_finest_units:
        if k < 1:
            continue
        sub = log_prices[::k]
        if len(sub) < 5:
            rows.append({'freq': k, 'rv': np.nan, 'rk': np.nan,
                          'tsrv': np.nan, 'n_bars': len(sub)})
            continue
        r = np.diff(sub)
        rv = simple_rv(r)
        rk = realized_kernel(r) if len(r) >= 20 else float('nan')
        tsrv = two_scale_rv(r, K=5) if len(r) >= 20 else float('nan')
        rows.append({'freq': k, 'rv': rv, 'rk': rk, 'tsrv': tsrv,
                      'n_bars': len(r)})
    return pd.DataFrame(rows)
