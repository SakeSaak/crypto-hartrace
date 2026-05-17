"""Parameter estimation for HAR-family models.

Implements OLS with HAC standard errors (β-inference) and Maximum
Likelihood under Hansen skewed-t innovations (for likelihood comparison
and innovation parameters ν, λ). Supports three σ_t dynamics:

    static  : σ_t = σ                                  (3 extra params: σ, ν, λ)
    gas     : log σ²_t = ω + A·s_{t-1} + B·log σ²_{t-1} (6 extra params)
    egarch  : log σ²_t = ω + α(|z|-E|z|) + δ·z + β·log σ²_{t-1} (7 extra)

Families covered (no exogenous X yet — they need external data):

    HAR        baseline Corsi cascade on log RV
    HAR-Q      + measurement-error correction
    HAR-J      baseline cascade plus a single jump regressor
    HAR-CJ     continuous/jump-decomposed cascade
    HAR-CJ-Q   HAR-CJ + Q-correction on daily lag
    HAR-RS     semi-variance daily lag + RV weekly/monthly
    HAR-RS-Q   + Q-correction on both semi-variance daily lags
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from .distributions import loglik as skewt_loglik


# =====================================================================
# Feature construction
# =====================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cascade aggregates, log transforms, Q-proxy and target.

    Input columns required: date, RV, BPV, RQ, RS_neg, RS_pos, C, J.
    Output adds:
        log_RV, log_C, log_J (=log(1+J)), log_RS_neg, log_RS_pos
        *_D, *_W (5-day mean), *_M (22-day mean) cascade aggregates
        Q = √RQ/RV − sample mean
        log_*_D_Q = (daily) * Q interaction terms
        target = log_RV.shift(-1)
    """
    df = df.copy().sort_values('date').reset_index(drop=True)

    df['log_RV'] = np.log(df['RV'])
    df['log_C'] = np.log(df['C'].clip(lower=1e-12))
    df['log_J'] = np.log1p(df['J'])
    df['log_RS_neg'] = np.log(df['RS_neg'].clip(lower=1e-12))
    df['log_RS_pos'] = np.log(df['RS_pos'].clip(lower=1e-12))

    for pref in ('log_RV', 'log_C', 'log_J'):
        df[f'{pref}_D'] = df[pref]
        df[f'{pref}_W'] = df[pref].rolling(5).mean()
        df[f'{pref}_M'] = df[pref].rolling(22).mean()
    df['log_RS_neg_D'] = df['log_RS_neg']
    df['log_RS_pos_D'] = df['log_RS_pos']

    df['rq_proxy'] = np.sqrt(df['RQ']) / df['RV']
    df['Q'] = df['rq_proxy'] - df['rq_proxy'].mean()

    df['log_RV_D_Q'] = df['log_RV_D'] * df['Q']
    df['log_C_D_Q'] = df['log_C_D'] * df['Q']
    df['log_RS_neg_D_Q'] = df['log_RS_neg_D'] * df['Q']
    df['log_RS_pos_D_Q'] = df['log_RS_pos_D'] * df['Q']

    df['target'] = df['log_RV'].shift(-1)
    return df


FAMILY_COLS: Dict[str, List[str]] = {
    'HAR':      ['const', 'log_RV_D', 'log_RV_W', 'log_RV_M'],
    'HAR-Q':    ['const', 'log_RV_D', 'log_RV_D_Q', 'log_RV_W', 'log_RV_M'],
    'HAR-J':    ['const', 'log_RV_D', 'log_RV_W', 'log_RV_M', 'log_J_D'],
    'HAR-CJ':   ['const', 'log_C_D', 'log_C_W', 'log_C_M',
                 'log_J_D', 'log_J_W', 'log_J_M'],
    'HAR-CJ-Q': ['const', 'log_C_D', 'log_C_D_Q', 'log_C_W', 'log_C_M',
                 'log_J_D', 'log_J_W', 'log_J_M'],
    'HAR-RS':   ['const', 'log_RS_neg_D', 'log_RS_pos_D',
                 'log_RV_W', 'log_RV_M'],
    'HAR-RS-Q': ['const', 'log_RS_neg_D', 'log_RS_neg_D_Q',
                 'log_RS_pos_D', 'log_RS_pos_D_Q',
                 'log_RV_W', 'log_RV_M'],
}


def get_design_matrix(
    df_feat: pd.DataFrame, family: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, col_names) for a given family, dropping NaNs."""
    cols = FAMILY_COLS[family]
    keep = [c for c in cols if c != 'const']
    sub = df_feat[keep + ['target']].dropna()
    n = len(sub)
    X = np.column_stack([np.ones(n), sub[keep].values])
    y = sub['target'].values
    return X.astype(float), y.astype(float), list(cols)


# =====================================================================
# OLS with HAC (Newey-West, lag = 22)
# =====================================================================
def fit_ols_hac(X: np.ndarray, y: np.ndarray, col_names: List[str],
                 hac_lags: int = 22) -> dict:
    """OLS with Newey-West HAC standard errors."""
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    fitted = res.fittedvalues
    resid = res.resid
    return {
        'beta': np.asarray(res.params),
        'se': np.asarray(res.bse),
        't_stat': np.asarray(res.tvalues),
        'p_value': np.asarray(res.pvalues),
        'col_names': col_names,
        'sigma': float(np.std(resid, ddof=X.shape[1])),
        'r_squared': float(res.rsquared),
        'r_squared_adj': float(res.rsquared_adj),
        'n': int(len(y)), 'k': int(X.shape[1]),
        'fitted': fitted, 'residuals': resid,
    }


# =====================================================================
# ML: skewed-t with static σ
# =====================================================================
def _nll_skewt_static(theta, X, y):
    k = X.shape[1]
    beta = theta[:k]
    log_sigma = theta[k]
    nu = theta[k + 1]
    lam = theta[k + 2]
    if not (2.05 < nu < 80.0) or not (-0.98 < lam < 0.98):
        return 1e10
    sigma = float(np.exp(log_sigma))
    z = (y - X @ beta) / sigma
    try:
        ll = float(np.sum(skewt_loglik(z, nu, lam))) - len(y) * log_sigma
    except Exception:
        return 1e10
    return -ll


def fit_skewt_static(X, y, col_names, ols=None) -> dict:
    if ols is None:
        ols = fit_ols_hac(X, y, col_names)
    k = X.shape[1]
    init = np.concatenate([ols['beta'], [np.log(ols['sigma']), 8.0, -0.1]])
    res = minimize(_nll_skewt_static, init, args=(X, y),
                   method='L-BFGS-B',
                   options={'maxiter': 2000, 'gtol': 1e-7})
    beta = res.x[:k]
    log_sigma = float(res.x[k])
    sigma = float(np.exp(log_sigma))
    nu = float(res.x[k + 1])
    lam = float(res.x[k + 2])
    ll = -float(res.fun)
    n_par = k + 3
    n = len(y)
    return {
        'scale_type': 'static',
        'col_names': col_names,
        'beta': beta, 'sigma': sigma, 'nu': nu, 'lam': lam,
        'loglik': ll,
        'aic': 2 * n_par - 2 * ll,
        'bic': n_par * np.log(n) - 2 * ll,
        'n': n, 'n_params': n_par,
        'converged': bool(res.success),
        'residuals': y - X @ beta,
        'fitted': X @ beta,
        'sigma_path': np.full(n, sigma),
    }


# =====================================================================
# ML: skewed-t with GAS σ_t
# =====================================================================
def _gas_update(log_var_prev, z, omega, A, B, nu):
    """One-step update of log σ²_t using simplified t-score."""
    score = (nu + 1.0) * z * z / (nu - 2.0 + z * z) - 1.0
    return omega + A * score + B * log_var_prev


def _nll_skewt_gas(theta, X, y):
    k = X.shape[1]
    beta = theta[:k]
    log_var0 = theta[k]
    omega = theta[k + 1]
    A = theta[k + 2]
    B = theta[k + 3]
    nu = theta[k + 4]
    lam = theta[k + 5]
    if not (2.05 < nu < 80.0) or not (-0.98 < lam < 0.98):
        return 1e10
    if not (-0.5 < A < 0.5) or not (0.0 < B < 0.9995):
        return 1e10
    mu = X @ beta
    n = len(y)
    log_var = np.empty(n)
    log_var[0] = log_var0
    ll = 0.0
    for t in range(n):
        if log_var[t] > 50 or log_var[t] < -50:
            return 1e10
        sigma_t = float(np.exp(0.5 * log_var[t]))
        z = (y[t] - mu[t]) / sigma_t
        try:
            ll_t = float(skewt_loglik(np.array([z]), nu, lam)[0]) - 0.5 * log_var[t]
        except Exception:
            return 1e10
        ll += ll_t
        if t < n - 1:
            log_var[t + 1] = _gas_update(log_var[t], z, omega, A, B, nu)
    return -ll


def fit_skewt_gas(X, y, col_names, static_fit=None) -> dict:
    if static_fit is None:
        static_fit = fit_skewt_static(X, y, col_names)
    k = X.shape[1]
    # Warm-start
    init = np.concatenate([
        static_fit['beta'],
        [2 * np.log(static_fit['sigma']), 0.05, 0.05, 0.90,
         static_fit['nu'], static_fit['lam']],
    ])
    res = minimize(_nll_skewt_gas, init, args=(X, y),
                   method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
    beta = res.x[:k]
    log_var0 = float(res.x[k])
    omega = float(res.x[k + 1])
    A = float(res.x[k + 2])
    B = float(res.x[k + 3])
    nu = float(res.x[k + 4])
    lam = float(res.x[k + 5])
    # Replay path
    mu = X @ beta
    n = len(y)
    log_var = np.empty(n)
    log_var[0] = log_var0
    for t in range(n - 1):
        z = (y[t] - mu[t]) / float(np.exp(0.5 * log_var[t]))
        log_var[t + 1] = _gas_update(log_var[t], z, omega, A, B, nu)
    sigma_path = np.exp(0.5 * log_var)
    ll = -float(res.fun)
    n_par = k + 6
    return {
        'scale_type': 'gas', 'col_names': col_names,
        'beta': beta, 'omega': omega, 'A': A, 'B': B,
        'nu': nu, 'lam': lam, 'log_var0': log_var0,
        'loglik': ll, 'aic': 2 * n_par - 2 * ll,
        'bic': n_par * np.log(n) - 2 * ll,
        'n': n, 'n_params': n_par, 'converged': bool(res.success),
        'residuals': y - mu, 'fitted': mu, 'sigma_path': sigma_path,
    }


# =====================================================================
# ML: skewed-t with EGARCH σ_t
# =====================================================================
def _egarch_update(log_var_prev, z, omega, alpha, delta, beta_p, e_abs_z):
    return omega + alpha * (abs(z) - e_abs_z) + delta * z + beta_p * log_var_prev


def _e_abs_t(nu):
    """E|z| for variance-1 standardized Student-t with ν df."""
    from scipy.special import gammaln
    return float(np.exp(
        np.log(2.0) + 0.5 * np.log(nu - 2.0) + gammaln(0.5 * (nu + 1.0))
        - np.log(nu - 1.0) - 0.5 * np.log(np.pi) - gammaln(0.5 * nu)
    ))


def _nll_skewt_egarch(theta, X, y):
    k = X.shape[1]
    beta = theta[:k]
    log_var0 = theta[k]
    omega = theta[k + 1]
    alpha = theta[k + 2]
    delta = theta[k + 3]
    beta_p = theta[k + 4]
    nu = theta[k + 5]
    lam = theta[k + 6]
    if not (2.05 < nu < 80.0) or not (-0.98 < lam < 0.98):
        return 1e10
    if not (0.0 < beta_p < 0.9995) or not (-1.0 < alpha < 1.0):
        return 1e10
    if not (-0.5 < delta < 0.5):
        return 1e10
    e_abs = _e_abs_t(nu)
    mu = X @ beta
    n = len(y)
    log_var = np.empty(n)
    log_var[0] = log_var0
    ll = 0.0
    for t in range(n):
        if log_var[t] > 50 or log_var[t] < -50:
            return 1e10
        sigma_t = float(np.exp(0.5 * log_var[t]))
        z = (y[t] - mu[t]) / sigma_t
        try:
            ll_t = float(skewt_loglik(np.array([z]), nu, lam)[0]) - 0.5 * log_var[t]
        except Exception:
            return 1e10
        ll += ll_t
        if t < n - 1:
            log_var[t + 1] = _egarch_update(
                log_var[t], z, omega, alpha, delta, beta_p, e_abs
            )
    return -ll


def fit_skewt_egarch(X, y, col_names, static_fit=None) -> dict:
    if static_fit is None:
        static_fit = fit_skewt_static(X, y, col_names)
    k = X.shape[1]
    init = np.concatenate([
        static_fit['beta'],
        [2 * np.log(static_fit['sigma']), -0.05, 0.10, -0.05, 0.95,
         static_fit['nu'], static_fit['lam']],
    ])
    res = minimize(_nll_skewt_egarch, init, args=(X, y),
                   method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
    beta = res.x[:k]
    log_var0 = float(res.x[k])
    omega = float(res.x[k + 1])
    alpha = float(res.x[k + 2])
    delta = float(res.x[k + 3])
    beta_p = float(res.x[k + 4])
    nu = float(res.x[k + 5])
    lam = float(res.x[k + 6])
    e_abs = _e_abs_t(nu)
    mu = X @ beta
    n = len(y)
    log_var = np.empty(n)
    log_var[0] = log_var0
    for t in range(n - 1):
        z = (y[t] - mu[t]) / float(np.exp(0.5 * log_var[t]))
        log_var[t + 1] = _egarch_update(
            log_var[t], z, omega, alpha, delta, beta_p, e_abs
        )
    sigma_path = np.exp(0.5 * log_var)
    ll = -float(res.fun)
    n_par = k + 7
    return {
        'scale_type': 'egarch', 'col_names': col_names,
        'beta': beta, 'omega': omega, 'alpha': alpha, 'delta': delta,
        'beta_p': beta_p, 'nu': nu, 'lam': lam, 'log_var0': log_var0,
        'loglik': ll, 'aic': 2 * n_par - 2 * ll,
        'bic': n_par * np.log(n) - 2 * ll,
        'n': n, 'n_params': n_par, 'converged': bool(res.success),
        'residuals': y - mu, 'fitted': mu, 'sigma_path': sigma_path,
    }


# =====================================================================
# Diagnostics on residuals
# =====================================================================
def residual_diagnostics(residuals: np.ndarray, sigma_path: np.ndarray) -> dict:
    """Standardized-residual diagnostics."""
    z = residuals / sigma_path
    z = z[np.isfinite(z)]
    out = {}
    for lg in (5, 10, 22):
        lb = acorr_ljungbox(z, lags=[lg], return_df=True)
        out[f'LB_{lg}_p'] = float(lb['lb_pvalue'].iloc[0])
    for lg in (5, 10, 22):
        lm, lm_p, _, _ = het_arch(z, nlags=lg)
        out[f'ARCH_{lg}_p'] = float(lm_p)
    out['mean_z'] = float(z.mean())
    out['std_z'] = float(z.std(ddof=1))
    out['skew_z'] = float(pd.Series(z).skew())
    out['kurt_z'] = float(pd.Series(z).kurt())  # excess
    return out
