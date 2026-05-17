"""E6 — Cross-asset replication of HAR-RS-DOW dominance.

Simpler version: in-sample BIC + simple OOS evaluation (no VaR), per asset.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_static

PROJECT = Path(__file__).parent.parent
TBL = PROJECT / 'outputs' / 'tables'

ASSETS = [
    ('BTC-EUR', 'realized_measures_btc_eur_v2.parquet'),
    ('ETH-EUR', 'realized_measures_eth_eur_v2.parquet'),
    ('SOL-EUR', 'realized_measures_sol_eur_v2.parquet'),
]

def crps_normal(mu, sigma, y):
    """Closed-form CRPS for Normal forecast (good approximation for skewed-t when nu high)."""
    from scipy.stats import norm
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))


def run_asset(asset_name: str, rv_file: str):
    rv_path = PROJECT / 'data' / 'processed' / rv_file
    if not rv_path.exists():
        print(f"  {asset_name}: SKIP (no {rv_path.name})")
        return None
    
    rv = pd.read_parquet(rv_path)
    print(f"\n=== {asset_name} ===")
    print(f"  RV data: {len(rv)} days, {rv['date'].min().date()} → {rv['date'].max().date()}")
    
    feats = build_features_v2(rv_path)
    feats['const'] = 1.0
    n = len(feats)
    print(f"  Features built: {len(feats)} usable days")
    
    har_rs_dow_cols = FAMILIES_V2['HAR-RS-DOW']
    
    # Drop rijen met NaN (rolling-mean burn-in + last row target)
    feats = feats.dropna(subset=har_rs_dow_cols + ['target']).reset_index(drop=True)
    n = len(feats)
    print(f"  Na dropna: {n} usable days")
    
    # In-sample fit (op 67% van data)
    split = int(n * 0.67)
    X_train = feats[har_rs_dow_cols].iloc[:split].values
    y_train = feats['target'].iloc[:split].values
    
    fit = fit_skewt_static(X_train, y_train, har_rs_dow_cols)
    bic_is = fit['bic']
    
    # Simple OOS evaluation (no walkforward refit, single static fit)
    X_test = feats[har_rs_dow_cols].iloc[split:].values
    y_test = feats['target'].iloc[split:].values
    
    mu_pred = X_test @ fit['beta']
    sigma_resid = fit['sigma']
    crps_vals = crps_normal(mu_pred, sigma_resid, y_test)
    
    oos_crps = float(crps_vals.mean())
    oos_rmse = float(np.sqrt(((y_test - mu_pred) ** 2).mean()))
    oos_qlike = float((y_test - mu_pred + np.exp(y_test - mu_pred) - 1).mean())  # QLIKE in log
    y_mean = y_train.mean()
    oos_r2 = float(1 - ((y_test - mu_pred) ** 2).sum() / ((y_test - y_mean) ** 2).sum())
    
    # Parameter extraction
    col_names = har_rs_dow_cols
    beta = fit['beta']
    
    # Find RS+ en RS- parameter indices
    rs_pos_idx = next((i for i, c in enumerate(col_names) if 'RS_pos' in c.lower() or 'rs_pos' in c.lower()), None)
    rs_neg_idx = next((i for i, c in enumerate(col_names) if 'RS_neg' in c.lower() or 'rs_neg' in c.lower()), None)
    
    beta_d_plus = beta[rs_pos_idx] if rs_pos_idx is not None else np.nan
    beta_d_minus = beta[rs_neg_idx] if rs_neg_idx is not None else np.nan
    
    # DOW dummy params — find dow_1..dow_6
    dow_betas = {}
    for i, c in enumerate(col_names):
        for d in ['dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']:
            if d == c.lower() or c.lower().endswith(d):
                dow_betas[d] = beta[i]
    
    # Find Saturday (dow=5) en Monday (dow=0 or dow_1 in dummy coding)
    gamma_sat = dow_betas.get('dow_5', np.nan)
    gamma_mon = dow_betas.get('dow_1', np.nan)
    
    result = {
        'asset': asset_name,
        'n_total': n,
        'n_train': split,
        'n_oos': n - split,
        'bic_in_sample': float(bic_is),
        'oos_crps': oos_crps,
        'oos_qlike': oos_qlike,
        'oos_rmse': oos_rmse,
        'oos_r2': oos_r2,
        'beta_d_plus': float(beta_d_plus) if not np.isnan(beta_d_plus) else None,
        'beta_d_minus': float(beta_d_minus) if not np.isnan(beta_d_minus) else None,
        'leverage_asymmetry': float(beta_d_minus - beta_d_plus) if not (np.isnan(beta_d_plus) or np.isnan(beta_d_minus)) else None,
        'gamma_saturday': float(gamma_sat) if not np.isnan(gamma_sat) else None,
        'gamma_monday': float(gamma_mon) if not np.isnan(gamma_mon) else None,
        'nu_hat': float(fit.get('nu', np.nan)),
        'lambda_hat': float(fit.get('lam', np.nan)),
    }
    
    print(f"  In-sample BIC:  {bic_is:.0f}")
    print(f"  OOS CRPS:       {oos_crps:.4f}")
    print(f"  OOS R²:         {oos_r2:+.3f}")
    print(f"  β_d⁻ - β_d⁺:    {result['leverage_asymmetry']}")
    print(f"  γ_Sat (DOW):    {result['gamma_saturday']}")
    return result


if __name__ == "__main__":
    results = []
    for asset, rv_file in ASSETS:
        r = run_asset(asset, rv_file)
        if r:
            results.append(r)
    
    df = pd.DataFrame(results)
    df.to_csv(TBL / 'E6_cross_asset_summary.csv', index=False)
    
    print(f"\n{'='*100}")
    print("Cross-asset HAR-RS-DOW summary")
    print('='*100)
    print(f"{'Asset':<10}{'n_OOS':>7}{'BIC_IS':>10}{'CRPS':>9}{'R²':>8}{'β_d⁻-β_d⁺':>12}{'γ_Sat':>10}{'ν̂':>7}")
    print('-'*100)
    for _, r in df.iterrows():
        gamma_sat_str = f"{r['gamma_saturday']:+.3f}" if r['gamma_saturday'] is not None else 'n/a'
        lev_str = f"{r['leverage_asymmetry']:+.3f}" if r['leverage_asymmetry'] is not None else 'n/a'
        print(f"  {r['asset']:<8}{r['n_oos']:>7.0f}{r['bic_in_sample']:>10.0f}"
              f"{r['oos_crps']:>9.4f}{r['oos_r2']:>+8.3f}{lev_str:>12}{gamma_sat_str:>10}{r['nu_hat']:>7.2f}")
    print(f"\n✓ Saved: outputs/tables/E6_cross_asset_summary.csv")
