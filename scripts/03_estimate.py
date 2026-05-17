"""Fit the HAR-family horse race on BTC-EUR realized measures.

For each family × scale-type combination:
  1. OLS with Newey-West HAC (lag = 22) — for β-inference
  2. ML with Hansen skew-t innovations — for log-lik / AIC / BIC and (ν, λ)
  3. ML with skew-t + GAS σ_t                        }  comparison via LR/AIC
  4. ML with skew-t + EGARCH σ_t                     }

Outputs
-------
  outputs/tables/03_ols_coefficients.csv    OLS-HAC betas per model
  outputs/tables/03_ml_summary.csv          AIC/BIC/loglik per (family, scale)
  outputs/tables/03_residual_diagnostics.csv
  outputs/03_estimation_summary.md
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.estimation import (
    build_features, get_design_matrix, FAMILY_COLS,
    fit_ols_hac, fit_skewt_static, fit_skewt_gas, fit_skewt_egarch,
    residual_diagnostics,
)

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
INPUT = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_15m.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / '03_estimation_summary.md'


def fmt_p(p):
    if not np.isfinite(p):
        return "  -  "
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def main():
    print("=" * 70)
    print("03_estimate.py — HAR-family horse race on BTC-EUR")
    print("=" * 70)

    print("\n[1/4] Building features")
    df = pd.read_parquet(INPUT)
    feat = build_features(df)
    print(f"  N raw    : {len(df)}")
    print(f"  N usable : {feat.dropna(subset=['target', 'log_RV_M']).shape[0]} "
          f"(loss = first 22 days for M-cascade + last 1 for target shift)")

    # --- OLS fits ---
    print("\n[2/4] OLS + HAC (lag=22) per family")
    ols_results = {}
    rows_ols = []
    for fam in FAMILY_COLS:
        X, y, names = get_design_matrix(feat, fam)
        r = fit_ols_hac(X, y, names)
        ols_results[fam] = r
        for j, nm in enumerate(names):
            rows_ols.append({
                'family': fam, 'param': nm,
                'beta': r['beta'][j], 'se_hac': r['se'][j],
                't_stat': r['t_stat'][j], 'p_value': r['p_value'][j],
            })
        sig = ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
               for p in r['p_value']]
        print(f"\n  {fam}: R² = {r['r_squared']:.4f}, σ̂ = {r['sigma']:.4f}, n = {r['n']}")
        for j, nm in enumerate(names):
            print(f"    {nm:<18} = {r['beta'][j]:+8.4f}  "
                  f"(HAC se {r['se'][j]:.4f},  t = {r['t_stat'][j]:+5.2f},  p = {fmt_p(r['p_value'][j])}) {sig[j]}")
    pd.DataFrame(rows_ols).to_csv(TBL / '03_ols_coefficients.csv', index=False)

    # --- ML fits ---
    print("\n[3/4] Maximum likelihood: skewed-t × {static, gas, egarch}")
    rows_ml = []
    rows_diag = []
    fits = {}
    for fam in FAMILY_COLS:
        X, y, names = get_design_matrix(feat, fam)
        ols = ols_results[fam]
        # 1. Static
        t0 = time.time()
        st = fit_skewt_static(X, y, names, ols=ols)
        dt_st = time.time() - t0
        # 2. GAS
        t0 = time.time()
        gas = fit_skewt_gas(X, y, names, static_fit=st)
        dt_gas = time.time() - t0
        # 3. EGARCH
        t0 = time.time()
        eg = fit_skewt_egarch(X, y, names, static_fit=st)
        dt_eg = time.time() - t0
        fits[fam] = {'static': st, 'gas': gas, 'egarch': eg}
        for name, f, dt in [('static', st, dt_st),
                              ('gas', gas, dt_gas),
                              ('egarch', eg, dt_eg)]:
            diag = residual_diagnostics(f['residuals'], f['sigma_path'])
            rows_ml.append({
                'family': fam, 'scale': name,
                'n': f['n'], 'n_params': f['n_params'],
                'loglik': f['loglik'], 'aic': f['aic'], 'bic': f['bic'],
                'sigma_mean': float(np.mean(f['sigma_path'])),
                'nu': f['nu'], 'lam': f['lam'],
                'converged': f['converged'], 'fit_sec': dt,
            })
            rows_diag.append({'family': fam, 'scale': name, **diag})
        print(f"  {fam:<10} static: ll = {st['loglik']:+8.2f}, AIC = {st['aic']:8.2f}, BIC = {st['bic']:8.2f}  "
              f"| GAS: ll = {gas['loglik']:+8.2f}, AIC = {gas['aic']:8.2f}  "
              f"| EGARCH: ll = {eg['loglik']:+8.2f}, AIC = {eg['aic']:8.2f}")

    ml_df = pd.DataFrame(rows_ml)
    ml_df.to_csv(TBL / '03_ml_summary.csv', index=False)
    diag_df = pd.DataFrame(rows_diag)
    diag_df.to_csv(TBL / '03_residual_diagnostics.csv', index=False)

    # --- Ranking ---
    print("\n[4/4] Model ranking by BIC (across all family×scale)")
    rank = ml_df.sort_values('bic').reset_index(drop=True)
    rank['ΔBIC'] = rank['bic'] - rank['bic'].min()
    rank_disp = rank[['family', 'scale', 'n_params', 'loglik', 'aic', 'bic', 'ΔBIC',
                       'nu', 'lam', 'sigma_mean', 'converged']]
    with pd.option_context('display.float_format', '{:.3f}'.format,
                            'display.width', 160, 'display.max_rows', 30):
        print(rank_disp.to_string(index=False))

    print("\n  Best by BIC: {} / {}".format(rank.iloc[0]['family'], rank.iloc[0]['scale']))
    print("  Best by AIC: {} / {}".format(ml_df.sort_values('aic').iloc[0]['family'],
                                           ml_df.sort_values('aic').iloc[0]['scale']))

    # --- Markdown summary ---
    md = ["# HAR horse race — parameter estimation\n"]
    md.append(f"Sample: {df['date'].min().date() if 'date' in df.columns else 'N/A'} "
               f"→ end (n usable = {st['n']})\n")
    md.append("## OLS-HAC coefficient highlights (significance @ 5%)\n")
    sig_summary = []
    for fam in FAMILY_COLS:
        r = ols_results[fam]
        for j, nm in enumerate(r['col_names']):
            if r['p_value'][j] < 0.05 and nm != 'const':
                sig_summary.append(f"- **{fam}** {nm}: β̂ = {r['beta'][j]:+.4f} "
                                    f"(HAC p = {fmt_p(r['p_value'][j])})")
    md.extend(sig_summary[:30])
    md.append("\n## ML ranking by BIC\n")
    md.append(rank_disp.head(15).to_markdown(index=False, floatfmt=".3f"))
    md.append("\n## Residual diagnostics (standardized residuals)\n")
    md.append("Lower p-values = remaining structure not captured by the model.\n")
    diag_pivot = diag_df.set_index(['family', 'scale'])[
        ['LB_22_p', 'ARCH_22_p', 'skew_z', 'kurt_z']
    ]
    md.append(diag_pivot.to_markdown(floatfmt=".3f"))
    md.append("")
    SUM.parent.mkdir(parents=True, exist_ok=True)
    SUM.write_text("\n".join(str(x) for x in md))
    print(f"\nSaved summary to {SUM.relative_to(PROJECT)}")
    print(f"Tables in     {TBL.relative_to(PROJECT)}/03_*.csv")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
