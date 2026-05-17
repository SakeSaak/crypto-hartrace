"""Stylized-facts diagnostics for the BTC-EUR realized-measures dataset.

Implements four families of tests that drive the model-component
choices for the HAR-CJ-X-Q and HAR-RS-X-Q horse race:

  1. Distribution properties     — moments, JB, KS, QQ-data
  2. Heavy tails / tail index    — Hill estimator, GPD fit, conditional moments
  3. Autocorrelation structure   — ACF/PACF, Ljung-Box, ARCH-LM
  4. Long memory                 — R/S analysis, GPH log-periodogram

Each function is dependency-light and returns dicts/tuples that the
script and notebook both consume.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genpareto, jarque_bera, kstest, norm, t as student_t
from statsmodels.tsa.stattools import acf as _sm_acf, pacf as _sm_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


# =====================================================================
# 1) Distribution diagnostics
# =====================================================================
def moments(x: np.ndarray) -> Dict[str, float]:
    """Return n, mean, std, skewness, excess kurtosis, JB, KS-vs-Normal."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    sk = float(stats.skew(x, bias=False))
    ek = float(stats.kurtosis(x, fisher=True, bias=False))  # excess kurtosis
    jb_stat, jb_p = jarque_bera(x)
    ks_stat, ks_p = kstest((x - mu) / sd, 'norm')
    return {
        'n': n, 'mean': mu, 'std': sd, 'skew': sk, 'excess_kurt': ek,
        'kurtosis': ek + 3.0,
        'jb_stat': float(jb_stat), 'jb_pvalue': float(jb_p),
        'ks_stat': float(ks_stat), 'ks_pvalue': float(ks_p),
        'q01': float(np.quantile(x, 0.01)),
        'q05': float(np.quantile(x, 0.05)),
        'q95': float(np.quantile(x, 0.95)),
        'q99': float(np.quantile(x, 0.99)),
        'min': float(x.min()), 'max': float(x.max()),
    }


def fit_student_t(x: np.ndarray) -> Dict[str, float]:
    """ML-fit a symmetric Student-t to standardized x. Returns ν, μ, σ."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nu, mu, sigma = student_t.fit(x)
    return {'nu': float(nu), 'mu': float(mu), 'sigma': float(sigma)}


def fit_hansen_skewt(x: np.ndarray) -> Dict[str, float]:
    """ML-fit Hansen skewed-t to standardized x. Returns ν, λ.

    Optimises the negative log-likelihood numerically via scipy.minimize.
    """
    from scipy.optimize import minimize
    from .distributions import loglik  # Hansen skewed-t log-density

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    z = (x - x.mean()) / x.std(ddof=1)

    def nll(theta):
        nu, lam = theta
        if nu <= 2.05 or abs(lam) >= 0.99:
            return 1e10
        try:
            return -float(np.sum(loglik(z, nu, lam)))
        except Exception:
            return 1e10

    res = minimize(
        nll, x0=[8.0, -0.05], method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 2000},
    )
    nu, lam = res.x
    return {'nu': float(nu), 'lam': float(lam), 'loglik': -float(res.fun),
            'converged': bool(res.success)}


def qq_points(
    x: np.ndarray,
    dist: str = 'norm',
    params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (theoretical_quantiles, sample_quantiles) for a QQ plot.

    `dist` ∈ {'norm', 't', 'skewt'}. For 'skewt', params={'nu':..., 'lam':...}.
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    p = (np.arange(1, n + 1) - 0.5) / n   # plotting positions
    if dist == 'norm':
        theo = norm.ppf(p)
    elif dist == 't':
        nu = (params or {}).get('nu', 5.0)
        theo = student_t.ppf(p, df=nu)
        # rescale to unit variance
        theo = theo / np.sqrt(nu / (nu - 2.0))
    elif dist == 'skewt':
        from .distributions import ppf as skewt_ppf
        nu = (params or {}).get('nu', 8.0)
        lam = (params or {}).get('lam', 0.0)
        theo = skewt_ppf(p, nu, lam)
    else:
        raise ValueError(f"Unknown dist: {dist}")
    # Standardize sample for comparable scale
    sample = (x - x.mean()) / x.std(ddof=1)
    return theo, sample


# =====================================================================
# 2) Tail / heavy-tail diagnostics
# =====================================================================
def hill_estimator(
    x: np.ndarray,
    k_range: Optional[Sequence[int]] = None,
    side: str = 'two-sided',
) -> Tuple[np.ndarray, np.ndarray]:
    """Hill estimator of the tail index along k order statistics.

    For the right tail of |x| with k largest order statistics:
        ξ̂(k) = (1/k) Σ_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)})
    The tail index α = 1/ξ. A finite-variance tail has α > 2 (ξ < 0.5).
    """
    if side == 'two-sided':
        v = np.abs(np.asarray(x, dtype=float))
    elif side == 'right':
        v = np.asarray(x, dtype=float)
    elif side == 'left':
        v = -np.asarray(x, dtype=float)
    else:
        raise ValueError(side)
    v = v[np.isfinite(v) & (v > 0)]
    v.sort()
    v = v[::-1]  # descending
    n = len(v)
    if k_range is None:
        k_range = np.arange(5, max(6, n // 4))
    k_arr = np.array(k_range, dtype=int)
    xi = np.full(len(k_arr), np.nan)
    for i, k in enumerate(k_arr):
        if 2 <= k < n:
            xi[i] = float(np.mean(np.log(v[:k]) - np.log(v[k])))
    return k_arr, xi


def gpd_tail(
    x: np.ndarray,
    threshold: Optional[float] = None,
    quantile: float = 0.95,
) -> Dict[str, float]:
    """Fit a Generalized Pareto Distribution to absolute exceedances.

    threshold u defaults to the `quantile`-th quantile of |x|.
    Returns shape ξ (positive ξ ⇒ heavy tail) and scale β.
    """
    v = np.abs(np.asarray(x, dtype=float))
    v = v[np.isfinite(v)]
    if threshold is None:
        threshold = float(np.quantile(v, quantile))
    excess = v[v > threshold] - threshold
    if len(excess) < 30:
        return {'threshold': threshold, 'n_exc': len(excess),
                'xi': float('nan'), 'beta': float('nan'),
                'mean_exc': float('nan')}
    xi, _, beta = genpareto.fit(excess, floc=0.0)
    return {'threshold': threshold, 'n_exc': len(excess),
            'xi': float(xi), 'beta': float(beta),
            'mean_exc': float(excess.mean())}


# =====================================================================
# 3) Autocorrelation / clustering diagnostics
# =====================================================================
def acf_with_ci(
    x: np.ndarray, nlags: int = 40, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """ACF up to lag `nlags` with (alpha)-confidence bands. Returns (ac, ci)."""
    ac, ci = _sm_acf(x, nlags=nlags, alpha=alpha, fft=True, missing='drop')
    return np.asarray(ac), np.asarray(ci)


def pacf_with_ci(
    x: np.ndarray, nlags: int = 40, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    pc, ci = _sm_pacf(x, nlags=nlags, alpha=alpha, method='ywm')
    return np.asarray(pc), np.asarray(ci)


def ljung_box(
    x: np.ndarray, lags: Sequence[int] = (5, 10, 22),
) -> pd.DataFrame:
    """Ljung-Box Q-statistic and p-values at requested lags."""
    return acorr_ljungbox(np.asarray(x), lags=list(lags), return_df=True)


def arch_lm(
    x: np.ndarray, nlags: int = 5,
) -> Dict[str, float]:
    """Engle's ARCH-LM test for conditional heteroskedasticity."""
    lm, lm_p, f, f_p = het_arch(np.asarray(x), nlags=nlags)
    return {'lags': nlags, 'lm_stat': float(lm), 'lm_pvalue': float(lm_p),
            'f_stat': float(f), 'f_pvalue': float(f_p)}


# =====================================================================
# 4) Long-memory diagnostics
# =====================================================================
def hurst_rs(
    x: np.ndarray,
    min_chunk: int = 10,
    max_chunk: Optional[int] = None,
    n_chunks: int = 25,
) -> Dict[str, Any]:
    """Hurst exponent via R/S analysis.

    H = 0.5     : no long memory (random walk increments / Brownian)
    H > 0.5     : positive long-range dependence (persistence)
    H < 0.5     : anti-persistence
    Empirical equity vol typically H ≈ 0.75–0.85.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if max_chunk is None:
        max_chunk = n // 2
    sizes = np.unique(
        np.geomspace(min_chunk, max_chunk, num=n_chunks).astype(int)
    )
    sizes = sizes[(sizes >= min_chunk) & (sizes <= n)]
    pts = []
    for c in sizes:
        n_seg = n // c
        if n_seg < 2:
            continue
        rs_vals = []
        for j in range(n_seg):
            seg = x[j * c:(j + 1) * c]
            y = np.cumsum(seg - seg.mean())
            r = y.max() - y.min()
            s = seg.std(ddof=1)
            if s > 0 and r > 0:
                rs_vals.append(r / s)
        if rs_vals:
            pts.append((c, float(np.mean(rs_vals))))
    sizes_used, rs_means = zip(*pts)
    log_n = np.log(sizes_used)
    log_rs = np.log(rs_means)
    slope, intercept = np.polyfit(log_n, log_rs, 1)
    # Standard error of the slope from the regression
    pred = intercept + slope * log_n
    resid = log_rs - pred
    sse = float(np.sum(resid ** 2))
    sxx = float(np.sum((log_n - log_n.mean()) ** 2))
    se = float(np.sqrt(sse / max(1, len(log_n) - 2) / sxx))
    return {'H': float(slope), 'se': se,
            'sizes': np.array(sizes_used), 'rs': np.array(rs_means),
            'log_n': log_n, 'log_rs': log_rs,
            'fit_intercept': float(intercept)}


def gph(x: np.ndarray, m: Optional[int] = None) -> Dict[str, float]:
    """Geweke-Porter-Hudak log-periodogram estimator of fractional d.

    Regress log I(ω_j) on log(4 sin²(ω_j/2)) for j = 1..m. d̂ is the
    negative slope. For a stationary long-memory process, d ∈ (0, 0.5);
    d > 0 indicates long memory.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    if m is None:
        m = int(np.power(n, 0.5))   # standard bandwidth: m = n^{1/2}
    fft = np.fft.fft(x)
    pgram = np.abs(fft[1:n // 2 + 1]) ** 2 / (2 * np.pi * n)
    freqs = 2 * np.pi * np.arange(1, n // 2 + 1) / n
    log_p = np.log(pgram[:m])
    reg = np.log(4.0 * np.sin(freqs[:m] / 2.0) ** 2)
    X = np.column_stack([np.ones(m), -reg])
    beta, *_ = np.linalg.lstsq(X, log_p, rcond=None)
    d_hat = float(beta[1])
    # Geweke-Porter-Hudak asymptotic SE: π/√(24m)
    se = float(np.pi / np.sqrt(24.0 * m))
    return {'d': d_hat, 'se': se, 'm': m,
            't_stat': d_hat / se, 'pvalue_two_sided': 2 * (1 - norm.cdf(abs(d_hat / se)))}


# =====================================================================
# 5) Extended diagnostics (v2)
# =====================================================================
def hurst_of_increments(x: np.ndarray, q_range=None) -> dict:
    """Generalised Hurst on log-vol *increments* (Takaishi 2020).

    The standard Hurst on log-vol *levels* is high (long memory in vol),
    but the Hurst on log-vol *increments* can be < 0.5 ("rough"). This
    matters for option pricing and for the rough-volatility literature
    (Gatheral-Jaisson-Rosenbaum 2018). We compute H(q) for q=1 (default).

    For a process Y(t) with stationary increments:
        E[ |Y(t+τ) − Y(t)|^q ]  ∝  τ^{q·H(q)}
    so a log-log regression on increments at multiple lags gives H(q).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if q_range is None:
        q_range = [1.0]
    out = {}
    lags = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50], dtype=int)
    lags = lags[lags < n // 4]
    if len(lags) < 3:
        return {f'H_q{q:.1f}': float('nan') for q in q_range}
    for q in q_range:
        log_tau = []
        log_moment = []
        for tau in lags:
            incs = np.abs(x[tau:] - x[:-tau])
            if len(incs) > 0:
                mq = float(np.mean(incs ** q))
                if mq > 0:
                    log_tau.append(np.log(tau))
                    log_moment.append(np.log(mq))
        if len(log_tau) >= 3:
            slope, _ = np.polyfit(log_tau, log_moment, 1)
            out[f'H_q{q:.1f}'] = float(slope / q)
        else:
            out[f'H_q{q:.1f}'] = float('nan')
    return out


def inverse_leverage_test(returns: np.ndarray, log_rv: np.ndarray) -> dict:
    """Test for the (inverse) leverage effect specifically for crypto.

    Regress log RV_{t+1} on:
      - log RV_t
      - r_{t}                           (signed return: continuous leverage)
      - 1{r_{t} < 0} · |r_{t}|         (negative-only effect, Engle-Ng style)
      - 1{r_{t} > 0} · |r_{t}|         (positive-only effect, the inverse-leverage tell)

    For Bitcoin, prior literature reports the *positive* coefficient is
    larger than the negative one (Bouri et al.) — inverse leverage.
    """
    import statsmodels.api as sm
    r = np.asarray(returns, dtype=float)
    y = np.asarray(log_rv, dtype=float)
    n = min(len(r), len(y))
    r = r[:n]
    y = y[:n]
    # Build design: lag log_rv + leverage interactions on lagged return
    X = np.column_stack([
        np.ones(n - 1),
        y[:-1],
        r[:-1],
        np.where(r[:-1] < 0, np.abs(r[:-1]), 0.0),
        np.where(r[:-1] > 0, np.abs(r[:-1]), 0.0),
    ])
    y_target = y[1:]
    model = sm.OLS(y_target, X).fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    names = ['const', 'log_RV_lag', 'r_lag', 'neg_abs', 'pos_abs']
    return {
        'names': names,
        'beta': np.asarray(model.params),
        'se': np.asarray(model.bse),
        'p_value': np.asarray(model.pvalues),
        't_stat': np.asarray(model.tvalues),
        'r_squared': float(model.rsquared),
        'n': int(n - 1),
    }


def periodicity_tests(log_rv: pd.Series, dow: pd.Series) -> dict:
    """Day-of-week and weekend-effect F-tests via OLS with dummies.

    H_0: all DOW means equal vs H_1: not all equal.
    Also reports per-DOW means.
    """
    import statsmodels.api as sm
    df = pd.DataFrame({'y': np.asarray(log_rv), 'dow': np.asarray(dow)})
    df = df.dropna()
    # Means per dow
    means = df.groupby('dow')['y'].mean().to_dict()
    # F-test: full vs restricted (intercept only)
    dummies = pd.get_dummies(df['dow'], prefix='dow',
                              drop_first=True, dtype=float)
    X_full = sm.add_constant(dummies.values)
    res_full = sm.OLS(df['y'].values, X_full).fit()
    X_rest = np.ones((len(df), 1))
    res_rest = sm.OLS(df['y'].values, X_rest).fit()
    # F-stat manually
    rss_full = float(((df['y'].values - res_full.fittedvalues) ** 2).sum())
    rss_rest = float(((df['y'].values - res_rest.fittedvalues) ** 2).sum())
    k_diff = X_full.shape[1] - 1   # 6 dow dummies
    df_full = len(df) - X_full.shape[1]
    F_stat = ((rss_rest - rss_full) / k_diff) / (rss_full / df_full)
    from scipy.stats import f as f_dist
    p_F = float(1.0 - f_dist.cdf(F_stat, k_diff, df_full))
    # Weekend effect: dummy = is_weekend
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    X_we = sm.add_constant(df['is_weekend'].values)
    res_we = sm.OLS(df['y'].values, X_we).fit(
        cov_type='HAC', cov_kwds={'maxlags': 22})
    return {
        'dow_means': means,
        'F_dow': float(F_stat),
        'p_F_dow': p_F,
        'k_diff': k_diff,
        'weekend_diff': float(res_we.params[1]),
        'weekend_se_hac': float(res_we.bse[1]),
        'weekend_p_hac': float(res_we.pvalues[1]),
    }


def engle_ng_sign_bias(residuals: np.ndarray, returns: np.ndarray) -> dict:
    """Engle-Ng (1993) sign-bias tests for asymmetric volatility.

    Regress squared standardized residuals on:
      const, 1{r_{t-1} < 0},  1{r_{t-1} < 0}·r_{t-1},  1{r_{t-1} >= 0}·r_{t-1}
    Joint F-test on the three slopes.
    """
    import statsmodels.api as sm
    e = np.asarray(residuals, dtype=float)
    r = np.asarray(returns, dtype=float)
    n = min(len(e), len(r))
    e = e[:n]; r = r[:n]
    e_std = e / e.std(ddof=1)
    e2 = e_std ** 2
    lag_neg = (r[:-1] < 0).astype(float)
    X = np.column_stack([
        np.ones(n - 1),
        lag_neg,                              # sign-bias
        lag_neg * r[:-1],                     # negative-size bias
        (1 - lag_neg) * r[:-1],               # positive-size bias
    ])
    model = sm.OLS(e2[1:], X).fit(cov_type='HAC', cov_kwds={'maxlags': 22})
    # Joint F-test on the three slope coefficients
    R = np.eye(4)[1:]   # rows 1,2,3
    f_test = model.f_test(R)
    return {
        'beta': np.asarray(model.params),
        'se': np.asarray(model.bse),
        'p_value': np.asarray(model.pvalues),
        'F_joint': float(f_test.fvalue),
        'p_F_joint': float(f_test.pvalue),
        'r_squared': float(model.rsquared),
        'n': int(n - 1),
    }
