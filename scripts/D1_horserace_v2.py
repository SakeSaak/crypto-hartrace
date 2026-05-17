"""Extended HAR horse race on v2 data (n≈2 300, log RK target).

Ten model variants:
    Tier 1 (baselines)        : HAR, HAR-CJ, HAR-RS
    Tier 2 (Q-correction)     : HAR-Q, HAR-RS-Q
    Tier 3 (periodicity, NEW) : HAR-WE, HAR-RS-WE, HAR-RS-DOW
    Tier 4 (macro/FX, NEW)    : HAR-RS-X
    Tier 5 (own hybrid)       : HAR-RS-Q-WE-X

Each variant fitted with:
    (a) OLS with HAC SE for β-inference
    (b) ML under Hansen skewed-t with static σ for likelihood comparison

Outputs:
    outputs/tables/D1_ols_v2.csv            OLS-HAC coefficients
    outputs/tables/D1_ml_v2.csv             AIC/BIC/loglik ranking
    outputs/D1_horserace_v2_summary.md
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_ols_hac, fit_skewt_static

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_INPUT = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_INPUT = PROJECT / 'data' / 'external' / 'eurusd.parquet'
MACRO_INPUT = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'D1_horserace_v2_summary.md'


def _get_xy(feat, family):
    cols = FAMILIES_V2[family]
    keep = [c for c in cols if c != 'const']
    missing = [c for c in keep if c not in feat.columns]
    if missing:
        raise RuntimeError(f"missing columns for {family}: {missing}")
    sub = feat[keep + ['target']].dropna()
    n = len(sub)
    X = np.column_stack([np.ones(n), sub[keep].values])
    return X.astype(float), sub['target'].values.astype(float), list(cols)


def _fmt_p(p):
    return f"{p:.3f}" if (not np.isfinite(p) or p >= 0.001) else f"{p:.1e}"


def main():
    print("=" * 72)
    print("D1_horserace_v2.py — extended HAR families on v2 data")
    print("=" * 72)

    print("\n[1/4] Building features")
    feat = build_features_v2(
        RV_INPUT, eurusd_path=FX_INPUT, fred_path=MACRO_INPUT,
        target='log_RK',
    )
    print(f"  n raw     : {len(feat)}")
    n_valid = feat[['target', 'log_RK_M']].dropna().shape[0]
    print(f"  n usable  : {n_valid}")
    print(f"  range     : {feat.index.min().date()} → {feat.index.max().date()}")
    print(f"  families  : {list(FAMILIES_V2.keys())}")

    # ===== OLS =====
    print("\n[2/4] OLS + HAC (lag=22) per family")
    ols_results = {}
    ols_rows = []
    for fam in FAMILIES_V2:
        X, y, names = _get_xy(feat, fam)
        r = fit_ols_hac(X, y, names)
        ols_results[fam] = r
        n_sig5 = int((r['p_value'] < 0.05).sum()) - 1  # excl intercept
        print(f"\n  {fam:<18} n={r['n']}  R²={r['r_squared']:.4f}  σ̂={r['sigma']:.4f}  "
              f"sig@5%: {n_sig5}/{r['k']-1} non-intercept")
        for j, nm in enumerate(names):
            ols_rows.append({'family': fam, 'param': nm,
                              'beta': r['beta'][j], 'se_hac': r['se'][j],
                              't': r['t_stat'][j], 'p': r['p_value'][j]})
            # Print significant ones inline
            if j > 0 and r['p_value'][j] < 0.05:
                star = '***' if r['p_value'][j] < 0.01 else '**'
                print(f"    {nm:<18} = {r['beta'][j]:+8.4f}  "
                      f"(p={_fmt_p(r['p_value'][j])}) {star}")
    pd.DataFrame(ols_rows).to_csv(TBL / 'D1_ols_v2.csv', index=False)

    # ===== ML skew-t static =====
    print("\n[3/4] ML skewed-t static σ per family")
    ml_rows = []
    for fam in FAMILIES_V2:
        X, y, names = _get_xy(feat, fam)
        t0 = time.time()
        st = fit_skewt_static(X, y, names, ols=ols_results[fam])
        dt = time.time() - t0
        ml_rows.append({
            'family': fam, 'n': st['n'], 'k': st['n_params'],
            'loglik': st['loglik'], 'aic': st['aic'], 'bic': st['bic'],
            'sigma': st['sigma'], 'nu': st['nu'], 'lam': st['lam'],
            'converged': st['converged'], 'fit_s': dt,
        })
        print(f"  {fam:<18}  ll={st['loglik']:+8.2f}  AIC={st['aic']:8.2f}  "
              f"BIC={st['bic']:8.2f}  ν̂={st['nu']:.2f}  λ̂={st['lam']:+.3f}  "
              f"({dt:.1f}s)")
    ml = pd.DataFrame(ml_rows)
    ml.to_csv(TBL / 'D1_ml_v2.csv', index=False)

    # ===== Ranking =====
    print("\n[4/4] Ranking by BIC")
    rank = ml.sort_values('bic').reset_index(drop=True)
    rank['ΔBIC'] = rank['bic'] - rank['bic'].min()
    cols = ['family', 'k', 'loglik', 'aic', 'bic', 'ΔBIC', 'sigma', 'nu', 'lam']
    with pd.option_context('display.float_format', '{:.2f}'.format,
                            'display.width', 160):
        print(rank[cols].to_string(index=False))

    # Markdown summary
    md = ["# D1 — extended HAR horse race (v2 data, n≈2 300)\n",
           f"Sample: {feat.index.min().date()} → {feat.index.max().date()} "
           f"({n_valid} usable)\n",
           "Target: log RK_{t+1} (5-min realized kernel)\n",
           "## ML ranking (BIC ascending)\n",
           rank[cols].to_markdown(index=False, floatfmt='.3f'),
           "\n## Significant OLS coefficients @ 5% (excl. intercept)\n",
           ]
    for fam in FAMILIES_V2:
        r = ols_results[fam]
        sig = [(nm, r['beta'][j], r['p_value'][j])
                for j, nm in enumerate(r['col_names'])
                if j > 0 and r['p_value'][j] < 0.05]
        md.append(f"\n### {fam}")
        for nm, b, p in sig:
            md.append(f"- `{nm}` = {b:+.4f}  (p = {_fmt_p(p)})")
    SUM.parent.mkdir(parents=True, exist_ok=True)
    SUM.write_text("\n".join(md))
    print(f"\nSaved: {SUM.relative_to(PROJECT)}")
    print(f"       {(TBL / 'D1_ols_v2.csv').relative_to(PROJECT)}")
    print(f"       {(TBL / 'D1_ml_v2.csv').relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
