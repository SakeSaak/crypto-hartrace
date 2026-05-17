"""Walk-forward backtest of the top-5 v2 HAR families, h=1.

Configuration
-------------
    train_window  : 1 000 days (rolling)
    refit_step    : 20 days
    horizon       : 1 day ahead
    target        : log RK_{t+1}
    families      : top-5 from D1 BIC ranking
        HAR-RS-DOW, HAR-RS-Q-WE-X, HAR-RS-X, HAR-RS-WE, HAR-WE

Outputs
-------
    outputs/tables/E1_walkforward_per_day_{family}.parquet
    outputs/tables/E1_walkforward_metrics.csv
    outputs/tables/E1_dm_matrix.csv
    outputs/E1_walkforward_summary.md
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_static
from hartrace.backtest import (
    walk_forward_h1, kupiec_test, christoffersen_test, diebold_mariano,
)

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E1_walkforward_summary.md'

TOP5 = ['HAR-RS-DOW', 'HAR-RS-Q-WE-X', 'HAR-RS-X', 'HAR-RS-WE', 'HAR-WE']
TRAIN_WIN = 1000
REFIT_STEP = 20
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)


def _fit(X, y, names):
    return fit_skewt_static(X, y, names)


def _aggregate_metrics(per_day: pd.DataFrame, family: str) -> dict:
    out = {'family': family,
           'n_test': len(per_day),
           'mean_crps': float(per_day['crps'].mean()),
           'median_crps': float(per_day['crps'].median()),
           'mean_qlike': float(per_day['qlike'].mean()),
           'median_qlike': float(per_day['qlike'].median()),
           'rmse_logrk': float(np.sqrt(((per_day['y_actual'] - per_day['mu_pred']) ** 2).mean())),
           'mae_logrk': float(np.abs(per_day['y_actual'] - per_day['mu_pred']).mean()),
           'r2_oos': float(1.0 - ((per_day['y_actual'] - per_day['mu_pred']) ** 2).sum()
                            / ((per_day['y_actual'] - per_day['y_actual'].mean()) ** 2).sum()),
           }
    # Coverage tests
    for a in VAR_ALPHAS:
        # log_RK violations
        v = per_day[f'viol_logrk_{int(a*100):02d}'].values
        kup = kupiec_test(v, a if a < 0.5 else 1 - a)
        out[f'rate_logrk_{int(a*100):02d}']    = kup['rate']
        out[f'p_kupiec_logrk_{int(a*100):02d}'] = kup['p_value']
        # return violations
        vr = per_day[f'viol_return_{int(a*100):02d}'].values
        kupr = kupiec_test(vr, a if a < 0.5 else 1 - a)
        out[f'rate_return_{int(a*100):02d}']    = kupr['rate']
        out[f'p_kupiec_return_{int(a*100):02d}'] = kupr['p_value']
        # Christoffersen joint
        ch = christoffersen_test(v, a if a < 0.5 else 1 - a)
        out[f'p_cc_logrk_{int(a*100):02d}'] = ch['p_cc']
        chr_ = christoffersen_test(vr, a if a < 0.5 else 1 - a)
        out[f'p_cc_return_{int(a*100):02d}'] = chr_['p_cc']
    return out


def main():
    print("=" * 72)
    print("E1_walkforward_v2.py — top-5 walk-forward backtest, h=1")
    print("=" * 72)

    print("\n[1/3] Features")
    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN,
                                target='log_RK')
    print(f"  n raw  : {len(feat)}")
    print(f"  range  : {feat.index.min().date()} → {feat.index.max().date()}")

    print(f"\n[2/3] Walk-forward per family (train_win={TRAIN_WIN}, refit={REFIT_STEP})")
    per_day = {}
    metrics = []
    for fam in TOP5:
        print(f"\n  --- {fam} ---")
        t0 = time.time()
        cols = FAMILIES_V2[fam]
        df_fam = walk_forward_h1(
            feat, family_cols=cols, fit_fn=_fit,
            train_window=TRAIN_WIN, refit_step=REFIT_STEP,
            var_alphas=VAR_ALPHAS, verbose=True,
        )
        dt = time.time() - t0
        out_path = TBL / f'E1_walkforward_per_day_{fam.replace("/", "_")}.parquet'
        df_fam.to_parquet(out_path)
        m = _aggregate_metrics(df_fam, fam)
        m['fit_time_s'] = round(dt, 1)
        metrics.append(m)
        per_day[fam] = df_fam
        print(f"  done in {dt:.1f}s, n_test={len(df_fam)}, "
              f"CRPS_mean={m['mean_crps']:.4f}, R²_oos={m['r2_oos']:+.4f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(TBL / 'E1_walkforward_metrics.csv', index=False)

    print("\n[3/3] Diebold-Mariano (mean QLIKE; negative ⇒ row beats column)")
    n_models = len(TOP5)
    dm_DM = pd.DataFrame(index=TOP5, columns=TOP5, dtype=float)
    dm_p = pd.DataFrame(index=TOP5, columns=TOP5, dtype=float)
    for i, fa in enumerate(TOP5):
        for j, fb in enumerate(TOP5):
            if i == j:
                dm_DM.iloc[i, j] = 0.0
                dm_p.iloc[i, j] = 1.0
                continue
            la = per_day[fa]['qlike'].values
            lb = per_day[fb]['qlike'].values
            dm = diebold_mariano(la, lb, h=1)
            dm_DM.iloc[i, j] = dm['DM_hln']
            dm_p.iloc[i, j] = dm['p_value']
    dm_DM.to_csv(TBL / 'E1_dm_qlike_matrix.csv')
    dm_p.to_csv(TBL / 'E1_dm_qlike_pvalues.csv')
    print("\nDM-HLN matrix (QLIKE; row vs column):")
    print(dm_DM.round(3).to_string())
    print("\nP-values:")
    print(dm_p.round(4).to_string())

    # Markdown summary
    md = ["# E1 — Walk-forward backtest, h=1 (top-5 v2 families)\n",
           f"Train window = {TRAIN_WIN}, refit every {REFIT_STEP} days. "
           f"Target = log RK_{{t+1}}.\n",
           "## Aggregate scores\n",
           metrics_df[['family', 'n_test', 'mean_crps', 'mean_qlike',
                        'rmse_logrk', 'r2_oos', 'fit_time_s']]
            .to_markdown(index=False, floatfmt='.4f'),
           "\n## VaR Kupiec rates (target = α)\n",
           ]
    var_table = metrics_df[['family'] + [f'rate_logrk_{int(a*100):02d}'
                                              for a in VAR_ALPHAS]]
    md.append("### log-RK VaR rates\n")
    md.append(var_table.to_markdown(index=False, floatfmt='.4f'))
    var_r = metrics_df[['family'] + [f'rate_return_{int(a*100):02d}'
                                          for a in VAR_ALPHAS]]
    md.append("\n### return VaR rates (using η ~ SkewT(ν=3, λ=0))\n")
    md.append(var_r.to_markdown(index=False, floatfmt='.4f'))
    md.append("\n## Diebold-Mariano (QLIKE, HLN-corrected)\n")
    md.append("DM matrix (row vs column; negative ⇒ row wins):\n")
    md.append(dm_DM.round(3).to_markdown())
    md.append("\nP-values:\n")
    md.append(dm_p.round(4).to_markdown())
    SUM.write_text("\n".join(md))
    print(f"\nSaved:")
    print(f"  → {SUM.relative_to(PROJECT)}")
    print(f"  → {(TBL / 'E1_walkforward_metrics.csv').relative_to(PROJECT)}")
    print(f"  → {(TBL / 'E1_dm_qlike_matrix.csv').relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
