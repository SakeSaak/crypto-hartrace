"""Walk-forward backtesting + density-forecast evaluation.

Forecast quality:
    crps_skewt_mc     CRPS via Monte Carlo from Hansen skewed-t
    qlike_logrv       QLIKE (Patton 2011) on log-RV scale

Coverage tests:
    kupiec_test       Unconditional coverage of VaR violations
    christoffersen    Conditional coverage + independence
    diebold_mariano   Pairwise forecast comparison (HLN-corrected)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as t_dist

from . import distributions as skewt


# ---------------------------------------------------------------------
# Density-forecast scoring
# ---------------------------------------------------------------------
def crps_skewt_mc(
    mu: float, sigma: float, nu: float, lam: float, y: float,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """CRPS for a skewed-t forecast distribution via Monte Carlo.

    CRPS(F, y) = E|X - y| - ½ E|X - X'|     (Gneiting-Raftery)
    """
    if rng is None:
        rng = np.random.default_rng()
    z = skewt.rvs(nu=nu, lam=lam, size=n_mc, rng=rng)
    samples = mu + sigma * z
    term1 = float(np.mean(np.abs(samples - y)))
    perm = rng.permutation(n_mc)
    term2 = float(np.mean(np.abs(samples - samples[perm])))
    return term1 - 0.5 * term2


def qlike_logrv(y_actual: float, y_pred: float) -> float:
    """QLIKE on log-RV (= QLIKE on RV after exponentiating).

    QLIKE(σ², σ̂²) = σ²/σ̂² − log(σ²/σ̂²) − 1
                  = exp(y − ŷ) − (y − ŷ) − 1            (in log scale)
    """
    d = y_actual - y_pred
    return float(np.exp(d) - d - 1.0)


# ---------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------
def kupiec_test(violations: np.ndarray, alpha: float) -> dict:
    """LR test for unconditional coverage of VaR violations (Kupiec 1995)."""
    v = np.asarray(violations).astype(int)
    n = len(v)
    x = int(v.sum())
    rate = x / n if n > 0 else 0.0
    if x in (0, n) or alpha in (0.0, 1.0):
        return {'n': n, 'violations': x, 'expected': n * alpha,
                 'rate': rate, 'alpha': alpha, 'LR': 0.0, 'p_value': 1.0}
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll1 = x * np.log(rate) + (n - x) * np.log(1 - rate)
    LR = float(2 * (ll1 - ll0))
    p = float(1.0 - chi2.cdf(LR, df=1))
    return {'n': n, 'violations': x, 'expected': n * alpha,
             'rate': rate, 'alpha': alpha, 'LR': LR, 'p_value': p}


def christoffersen_test(violations: np.ndarray, alpha: float) -> dict:
    """Conditional coverage = unconditional + independence (Christoffersen 1998)."""
    v = np.asarray(violations).astype(int)
    n = len(v)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())
    denom0 = n00 + n01
    denom1 = n10 + n11
    p01 = n01 / denom0 if denom0 > 0 else 0.0
    p11 = n11 / denom1 if denom1 > 0 else 0.0
    p_unc = (n01 + n11) / (n - 1) if n > 1 else 0.0

    def _llog(p):
        return np.log(p) if (p > 0 and p < 1) else 0.0

    if 0 < p_unc < 1 and (p01 + p11) > 0 and denom0 > 0 and denom1 > 0:
        ll_ind = (n00 * _llog(1 - p_unc) + n01 * _llog(p_unc)
                   + n10 * _llog(1 - p_unc) + n11 * _llog(p_unc))
        ll_alt = (n00 * _llog(1 - p01) + n01 * _llog(p01)
                   + n10 * _llog(1 - p11) + n11 * _llog(p11))
        LR_ind = float(max(0.0, 2 * (ll_alt - ll_ind)))
    else:
        LR_ind = 0.0
    p_ind = float(1.0 - chi2.cdf(LR_ind, df=1))

    kup = kupiec_test(v, alpha)
    LR_cc = kup['LR'] + LR_ind
    p_cc = float(1.0 - chi2.cdf(LR_cc, df=2))
    return {'LR_ind': LR_ind, 'p_ind': p_ind,
             'LR_cc': LR_cc, 'p_cc': p_cc,
             'kupiec': kup}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray,
                     h: int = 1) -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    Negative DM ⇒ A has lower average loss (A is better).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return {'DM': np.nan, 'DM_hln': np.nan, 'p_value': np.nan,
                 'n': n, 'mean_diff': np.nan}
    d_mean = float(np.mean(d))
    if h == 1:
        var_d = float(np.var(d, ddof=1)) / n
    else:
        lags = h - 1
        d_c = d - d_mean
        gamma = []
        for k in range(lags + 1):
            gamma.append(float(np.sum(d_c[k:] * d_c[:n - k]) / n))
        var_d = (gamma[0] + 2.0 * sum(
            (1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))) / n
    if var_d <= 0:
        return {'DM': np.nan, 'DM_hln': np.nan, 'p_value': np.nan,
                 'n': n, 'mean_diff': d_mean}
    dm = d_mean / np.sqrt(var_d)
    # HLN correction factor
    hln_factor = np.sqrt(max(1.0, (n + 1.0 - 2 * h + h * (h - 1) / n) / n))
    dm_hln = dm * hln_factor
    p_val = float(2.0 * (1.0 - t_dist.cdf(abs(dm_hln), df=n - 1)))
    return {'DM': float(dm), 'DM_hln': float(dm_hln),
             'p_value': p_val, 'n': n, 'mean_diff': d_mean}


# ---------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------
def walk_forward_h1(
    feat: pd.DataFrame,
    family_cols: list,
    fit_fn,
    target_col: str = 'target',
    train_window: int = 1000,
    refit_step: int = 20,
    var_alphas: tuple = (0.01, 0.05, 0.95, 0.99),
    eta_nu: float = 3.0,
    eta_lam: float = 0.0,
    n_mc_crps: int = 2000,
    rng_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-step-ahead walk-forward with periodic refit.

    Returns DataFrame: one row per OOS day with point forecast, density
    forecast diagnostics (CRPS, QLIKE), and VaR violations.
    """
    rng = np.random.default_rng(rng_seed)
    keep = [c for c in family_cols if c != 'const']
    cols_needed = keep + [target_col, 'r_d']
    df = feat[cols_needed].dropna().copy()
    n = len(df)
    if n < train_window + 50:
        raise RuntimeError(f"Not enough data: n={n} vs train_window={train_window}")

    fit_state = None
    rows = []
    for t in range(train_window, n):
        if (t - train_window) % refit_step == 0:
            sl = df.iloc[t - train_window:t]
            X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
            y_tr = sl[target_col].values.astype(float)
            fit_state = fit_fn(X_tr, y_tr, family_cols)
            if verbose and ((t - train_window) // refit_step) % 10 == 0:
                print(f"   refit @ t={t:4d}  ({(t-train_window)//refit_step:3d}); "
                      f"σ̂={fit_state['sigma']:.4f}, ν̂={fit_state['nu']:.2f}, "
                      f"λ̂={fit_state['lam']:+.3f}")
        beta = fit_state['beta']
        sigma = fit_state['sigma']
        nu = fit_state['nu']
        lam = fit_state['lam']

        # Features for day t → predict log_RK_{t+1}
        x_t = np.concatenate([[1.0], df.iloc[t][keep].values.astype(float)])
        mu_pred = float(np.dot(x_t, beta))
        y_actual = float(df.iloc[t][target_col])
        r_actual = float(df.iloc[t]['r_d'])

        # Density scoring
        crps = crps_skewt_mc(mu_pred, sigma, nu, lam, y_actual,
                              n_mc=n_mc_crps, rng=rng)
        qlike = qlike_logrv(y_actual, mu_pred)

        # VaR on log RK (upper-tail risk of high vol) using fitted skew-t
        row = {
            'date': df.index[t],
            'mu_pred': mu_pred, 'sigma_pred': sigma,
            'nu_pred': nu, 'lam_pred': lam,
            'y_actual': y_actual, 'r_actual': r_actual,
            'crps': crps, 'qlike': qlike,
        }
        for a in var_alphas:
            q_z = float(skewt.ppf(np.array([a]), nu=nu, lam=lam)[0])
            var_logrk = mu_pred + sigma * q_z
            row[f'var_logrk_{int(a*100):02d}'] = var_logrk
            row[f'viol_logrk_{int(a*100):02d}'] = int(y_actual > var_logrk) if a >= 0.5 \
                                                     else int(y_actual < var_logrk)

        # VaR on returns using exogenous return-innovation distribution η~SkewT(eta_nu, eta_lam)
        sigma_return = float(np.exp(mu_pred / 2.0))
        for a in var_alphas:
            q_z = float(skewt.ppf(np.array([a]), nu=eta_nu, lam=eta_lam)[0])
            var_r = sigma_return * q_z
            row[f'var_return_{int(a*100):02d}'] = var_r
            row[f'viol_return_{int(a*100):02d}'] = int(r_actual > var_r) if a >= 0.5 \
                                                      else int(r_actual < var_r)

        rows.append(row)

    return pd.DataFrame(rows).set_index('date')


# ---------------------------------------------------------------------
# Proper MC return-VaR (integrates over σ²_{t+1} predictive density)
# ---------------------------------------------------------------------
def return_var_proper_mc(
    mu_logrk: float, sigma_logrk: float,
    nu_resid: float, lam_resid: float,
    nu_eta: float, lam_eta: float,
    alphas: tuple,
    n_mc: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Proper MC integration for return-VaR.

    Plug-in: VaR_α(r) = sqrt(exp(μ̂_logRK)) · F_η⁻¹(α; ν_η, λ_η)
    Proper:  VaR_α(r) = α-quantile of
                  { sqrt(exp(μ̂ + σ_resid·z_resid)) · η  :  z_resid ~ skewT_resid,
                                                              η ~ skewT_η }

    The plug-in ignores E[σ²] vs Var[σ²] in the predictive density of
    log_RK. The proper one integrates over it. With M=5000 MC draws,
    the difference matters most in the deep tails.
    """
    if rng is None:
        rng = np.random.default_rng()
    z = skewt.rvs(nu=nu_resid, lam=lam_resid, size=n_mc, rng=rng)
    eta = skewt.rvs(nu=nu_eta, lam=lam_eta, size=n_mc, rng=rng)
    logrk_pred = mu_logrk + sigma_logrk * z
    # Cap log_RK to avoid overflow on extreme draws (cap at +20 in log space)
    sigma_ret = np.exp(0.5 * np.minimum(logrk_pred, 20.0))
    r_samples = sigma_ret * eta
    return {f'q{int(a*100):02d}': float(np.quantile(r_samples, a))
             for a in alphas}


# ---------------------------------------------------------------------
# Sigma-forward helpers (1-step ahead update for dynamic-σ models)
# ---------------------------------------------------------------------
def step_log_var_gas(log_var_prev: float, z: float,
                      omega: float, A: float, B: float, nu: float) -> float:
    """One-step-ahead log-variance update for GAS-skewT (Creal et al. 2013).

    Standard GAS score for skewed-t is the standardised
        (ν+1) · z² / (ν - 2 + z²)  −  1
    (using the symmetric-t approx — the skew-correction is negligible
    for our log-RK residuals where λ̂ ≈ 0.05).
    """
    score = (nu + 1.0) * z * z / (nu - 2.0 + z * z) - 1.0
    return (1.0 - B) * omega + B * log_var_prev + A * score


def step_log_var_egarch(log_var_prev: float, z: float,
                          omega: float, alpha: float, delta: float,
                          beta_p: float, e_abs_z: float) -> float:
    """One-step-ahead log-variance update for skewT-EGARCH (Nelson 1991).

        log σ²_t = ω + α·z_{t-1} + δ·(|z_{t-1}| − E|z|) + β_p·log σ²_{t-1}
    """
    return omega + alpha * z + delta * (abs(z) - e_abs_z) + beta_p * log_var_prev


# ---------------------------------------------------------------------
# Walk-forward with dynamic σ + proper MC return-VaR
# ---------------------------------------------------------------------
def walk_forward_dynamic(
    feat: pd.DataFrame,
    family_cols: list,
    fit_fn,                                 # static, gas, or egarch fitter
    sigma_model: str,                        # 'static' | 'gas' | 'egarch'
    target_col: str = 'target',
    train_window: int = 1000,
    refit_step: int = 20,
    var_alphas: tuple = (0.01, 0.05, 0.95, 0.99),
    eta_nu: float = 3.0,
    eta_lam: float = 0.0,
    n_mc: int = 5000,
    rng_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward with three optional σ dynamics + proper MC return-VaR.

    For dynamic σ models (gas/egarch) the σ² state is propagated between
    refits using the observed residuals; refit only re-estimates
    parameters but resets σ-state to fit's last log_var.
    """
    rng = np.random.default_rng(rng_seed)
    keep = [c for c in family_cols if c != 'const']
    df = feat[keep + [target_col, 'r_d']].dropna().copy()
    n = len(df)

    fit_state = None
    log_var_curr = None
    e_abs_z_cache = None     # for EGARCH; depends only on ν
    rows = []
    for t in range(train_window, n):
        if (t - train_window) % refit_step == 0:
            sl = df.iloc[t - train_window:t]
            X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
            y_tr = sl[target_col].values.astype(float)
            fit_state = fit_fn(X_tr, y_tr, family_cols)
            # Initialise σ-state to last in-sample log_var
            if sigma_model == 'static':
                log_var_curr = 2.0 * np.log(fit_state['sigma'])
            else:
                log_var_curr = 2.0 * np.log(fit_state['sigma_path'][-1])
            if sigma_model == 'egarch':
                # Pre-compute E|z| under SkewT(ν, 0)  (small-λ approx)
                from scipy.special import gammaln
                nu = fit_state['nu']
                e_abs_z_cache = float(np.exp(gammaln((nu + 1) / 2)
                                              - gammaln(nu / 2))
                                       * np.sqrt((nu - 2) / np.pi)
                                       * 2.0 / (nu - 1))
            if verbose and ((t - train_window) // refit_step) % 10 == 0:
                msg_params = (f"σ̂={np.exp(log_var_curr/2):.4f}, "
                                f"ν̂={fit_state['nu']:.2f}")
                print(f"   refit @ t={t:4d}  ({(t-train_window)//refit_step:3d}): {msg_params}")

        # Current 1-step-ahead density parameters
        beta = fit_state['beta']
        nu = fit_state['nu']
        lam = fit_state['lam']
        sigma_t = float(np.exp(log_var_curr / 2.0))

        x_t = np.concatenate([[1.0], df.iloc[t][keep].values.astype(float)])
        mu_t = float(np.dot(x_t, beta))
        y_t = float(df.iloc[t][target_col])
        r_t = float(df.iloc[t]['r_d'])

        crps = crps_skewt_mc(mu_t, sigma_t, nu, lam, y_t, n_mc=n_mc // 2, rng=rng)
        qlike = qlike_logrv(y_t, mu_t)

        # Log-RK VaR
        row = {'date': df.index[t], 'mu_pred': mu_t, 'sigma_pred': sigma_t,
               'nu_pred': nu, 'lam_pred': lam,
               'y_actual': y_t, 'r_actual': r_t,
               'crps': crps, 'qlike': qlike}
        for a in var_alphas:
            qz = float(skewt.ppf(np.array([a]), nu=nu, lam=lam)[0])
            v = mu_t + sigma_t * qz
            row[f'var_logrk_{int(a*100):02d}'] = v
            row[f'viol_logrk_{int(a*100):02d}'] = (int(y_t > v) if a >= 0.5
                                                       else int(y_t < v))

        # Proper MC return-VaR
        var_proper = return_var_proper_mc(mu_t, sigma_t, nu, lam,
                                            eta_nu, eta_lam, var_alphas,
                                            n_mc=n_mc, rng=rng)
        for a in var_alphas:
            v = var_proper[f'q{int(a*100):02d}']
            row[f'var_return_{int(a*100):02d}'] = v
            row[f'viol_return_{int(a*100):02d}'] = (int(r_t > v) if a >= 0.5
                                                       else int(r_t < v))

        # Step log_var forward for next iteration (dynamic σ only)
        if sigma_model != 'static':
            z_t = (y_t - mu_t) / sigma_t
            if sigma_model == 'gas':
                log_var_curr = step_log_var_gas(
                    log_var_curr, z_t,
                    fit_state['omega'], fit_state['A'], fit_state['B'],
                    nu)
            elif sigma_model == 'egarch':
                log_var_curr = step_log_var_egarch(
                    log_var_curr, z_t,
                    fit_state['omega'], fit_state['alpha'],
                    fit_state['delta'], fit_state['beta_p'],
                    e_abs_z_cache)

        rows.append(row)

    return pd.DataFrame(rows).set_index('date')


# ---------------------------------------------------------------------
# Dynamic-σ replay + walk-forward
# ---------------------------------------------------------------------
def _hansen_abc(nu, lam):
    """Hansen skewed-t a, b, c constants (also in distributions.py)."""
    from scipy.special import gammaln
    c = float(np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2) -
                       0.5 * np.log(np.pi * (nu - 2))))
    a = 4.0 * lam * c * (nu - 2) / (nu - 1)
    b = float(np.sqrt(1.0 + 3.0 * lam * lam - a * a))
    return a, b, c


def walk_forward_h1_dynamic(
    feat: pd.DataFrame,
    family_cols: list,
    fit_fn,
    scale_type: str,                 # 'gas' or 'egarch'
    target_col: str = 'target',
    train_window: int = 1000,
    refit_step: int = 20,
    var_alphas: tuple = (0.01, 0.05, 0.95, 0.99),
    eta_nu: float = 3.0,
    eta_lam: float = 0.0,
    n_mc_crps: int = 2000,
    n_mc_var: int = 4000,
    rng_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward with dynamic σ_t and MC-integrated return-VaR."""
    rng = np.random.default_rng(rng_seed)
    keep = [c for c in family_cols if c != 'const']
    df = feat[keep + [target_col, 'r_d']].dropna().copy()
    n = len(df)
    fit_state = None
    sigma_t = None
    last_train_log_var = None
    last_train_idx = None
    rows = []

    for t in range(train_window, n):
        if (t - train_window) % refit_step == 0:
            sl = df.iloc[t - train_window:t]
            X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
            y_tr = sl[target_col].values.astype(float)
            fit_state = fit_fn(X_tr, y_tr, family_cols)
            last_train_log_var = float(np.log(fit_state['sigma_path'][-1] ** 2))
            last_train_idx = t - 1
            # σ at last training row
            sigma_t = float(fit_state['sigma_path'][-1])
            # Replay one step forward using last training y
            y_lt = float(y_tr[-1])
            mu_lt = float(X_tr[-1] @ fit_state['beta'])
            z_lt = (y_lt - mu_lt) / sigma_t if sigma_t > 0 else 0.0
            if scale_type == 'gas':
                # Use the same GAS score form as in fit
                lv_new = (fit_state['omega']
                            + fit_state['A'] * ((fit_state['nu'] + 1) * z_lt ** 2
                                                / ((fit_state['nu'] - 2) + z_lt ** 2) - 1.0)
                            + fit_state['B'] * last_train_log_var)
            else:
                from .estimation import _e_abs_t
                e_abs = _e_abs_t(fit_state['nu'])
                lv_new = (fit_state['omega']
                            + fit_state['alpha'] * (abs(z_lt) - e_abs)
                            + fit_state['delta'] * z_lt
                            + fit_state['beta_p'] * last_train_log_var)
            sigma_t = float(np.exp(0.5 * lv_new))
            last_train_log_var = lv_new
            last_train_idx = t - 1
            if verbose and ((t - train_window) // refit_step) % 10 == 0:
                print(f"   refit @ t={t}: σ_pred at t={sigma_t:.4f}, "
                      f"ν̂={fit_state['nu']:.2f}, λ̂={fit_state['lam']:+.3f}")
        else:
            # Between refits: iterate σ forward using y_{t-1}
            y_prev = float(df.iloc[t - 1][target_col])
            x_prev = np.concatenate([[1.0],
                                       df.iloc[t - 1][keep].values.astype(float)])
            mu_prev = float(np.dot(x_prev, fit_state['beta']))
            z_prev = (y_prev - mu_prev) / sigma_t if sigma_t > 0 else 0.0
            if scale_type == 'gas':
                lv_new = (fit_state['omega']
                            + fit_state['A'] * ((fit_state['nu'] + 1) * z_prev ** 2
                                                / ((fit_state['nu'] - 2) + z_prev ** 2) - 1.0)
                            + fit_state['B'] * last_train_log_var)
            else:
                from .estimation import _e_abs_t
                e_abs = _e_abs_t(fit_state['nu'])
                lv_new = (fit_state['omega']
                            + fit_state['alpha'] * (abs(z_prev) - e_abs)
                            + fit_state['delta'] * z_prev
                            + fit_state['beta_p'] * last_train_log_var)
            sigma_t = float(np.exp(0.5 * lv_new))
            last_train_log_var = lv_new

        # Forecast for row t (target = y_{t+1})
        x_t = np.concatenate([[1.0], df.iloc[t][keep].values.astype(float)])
        mu_pred = float(np.dot(x_t, fit_state['beta']))
        sigma_pred = sigma_t
        nu = fit_state['nu']
        lam = fit_state['lam']
        y_actual = float(df.iloc[t][target_col])
        r_actual = float(df.iloc[t]['r_d'])

        # CRPS via MC
        crps = crps_skewt_mc(mu_pred, sigma_pred, nu, lam, y_actual,
                              n_mc=n_mc_crps, rng=rng)
        qlike = qlike_logrv(y_actual, mu_pred)

        row = {
            'date': df.index[t],
            'mu_pred': mu_pred, 'sigma_pred': sigma_pred,
            'nu_pred': nu, 'lam_pred': lam,
            'y_actual': y_actual, 'r_actual': r_actual,
            'crps': crps, 'qlike': qlike,
        }
        # VaR on log RK
        for a in var_alphas:
            q_z = float(skewt.ppf(np.array([a]), nu=nu, lam=lam)[0])
            var_lv = mu_pred + sigma_pred * q_z
            row[f'var_logrk_{int(a*100):02d}'] = var_lv
            row[f'viol_logrk_{int(a*100):02d}'] = int(y_actual > var_lv) if a >= 0.5 \
                                                     else int(y_actual < var_lv)

        # ---------- plug-in return-VaR (E1 method) ----------
        sigma_return_plugin = float(np.exp(mu_pred / 2.0))
        for a in var_alphas:
            q_z = float(skewt.ppf(np.array([a]), nu=eta_nu, lam=eta_lam)[0])
            var_r = sigma_return_plugin * q_z
            row[f'var_return_plug_{int(a*100):02d}'] = var_r
            row[f'viol_return_plug_{int(a*100):02d}'] = int(r_actual > var_r) if a >= 0.5 \
                                                          else int(r_actual < var_r)

        # ---------- MC-integrated return-VaR (vol-of-vol corrected) ----------
        # log_RK_{t+1}^* = mu_pred + sigma_pred * ε,  ε ~ SkewT(nu, lam)
        # r_{t+1}^* = exp(log_RK_{t+1}^* / 2) * η,    η ~ SkewT(eta_nu, eta_lam)
        eps = skewt.rvs(nu=nu, lam=lam, size=n_mc_var, rng=rng)
        eta = skewt.rvs(nu=eta_nu, lam=eta_lam, size=n_mc_var, rng=rng)
        logrk_sim = mu_pred + sigma_pred * eps
        r_sim = np.exp(logrk_sim / 2.0) * eta
        for a in var_alphas:
            q = float(np.quantile(r_sim, a))
            row[f'var_return_mc_{int(a*100):02d}'] = q
            row[f'viol_return_mc_{int(a*100):02d}'] = int(r_actual > q) if a >= 0.5 \
                                                        else int(r_actual < q)

        rows.append(row)

    return pd.DataFrame(rows).set_index('date')
