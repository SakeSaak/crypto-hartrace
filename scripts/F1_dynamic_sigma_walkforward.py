"""Three σ-variants of HAR-RS-DOW with proper MC return-VaR.

Isolates two sources of return-VaR miscalibration:
  (1) plug-in formula  vs proper MC integration over predictive σ²
  (2) static σ_resid   vs dynamic σ_resid (GAS / EGARCH)

For each of HAR-RS-DOW × {static, GAS, EGARCH}: walk-forward h=1,
train_win=1000, refit every 20 days. Compare CRPS, QLIKE, R²_oos
and VaR coverage of both log_RK upper-tail and *return* lower-tail.
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import (
    fit_skewt_static, fit_skewt_gas, fit_skewt_egarch,
)
from hartrace.backtest import (
    walk_forward_dynamic, kupiec_test, christoffersen_test, diebold_mariano,
)

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'F1_dynamic_sigma_summary.md'

FAMILY = 'HAR-RS-DOW'
SIGMA_MODELS = [
    ('static', fit_skewt_static),
    ('gas',    fit_skewt_gas),
    ('egarch', fit_skewt_egarch),
]
TRAIN_WIN = 1000
REFIT_STEP = 20
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)


def _aggregate(per_day, label):
    out = {'variant': label, 'n_test': len(per_day),
           'mean_crps': float(per_day['crps'].mean()),
           'mean_qlike': float(per_day['qlike'].mean()),
           'rmse_logrk': float(np.sqrt(((per_day['y_actual'] - per_day['mu_pred']) ** 2).mean())),
           'r2_oos': float(1.0 - ((per_day['y_actual'] - per_day['mu_pred']) ** 2).sum()
                            / ((per_day['y_actual'] - per_day['y_actual'].mean()) ** 2).sum()),
           'sigma_mean': float(per_day['sigma_pred'].mean()),
           'sigma_std':  float(per_day['sigma_pred'].std()),
           }
    for a in VAR_ALPHAS:
        # log-RK upper-tail
        v = per_day[f'viol_logrk_{int(a*100):02d}'].values
        kup = kupiec_test(v, a if a < 0.5 else 1 - a)
        out[f'rate_logrk_{int(a*100):02d}'] = kup['rate']
        out[f'p_kup_logrk_{int(a*100):02d}'] = kup['p_value']
        # return lower-tail
        v = per_day[f'viol_return_{int(a*100):02d}'].values
        kup = kupiec_test(v, a if a < 0.5 else 1 - a)
        out[f'rate_return_{int(a*100):02d}'] = kup['rate']
        out[f'p_kup_return_{int(a*100):02d}'] = kup['p_value']
    return out


def main():
    print("=" * 72)
    print(f"F1_dynamic_sigma_walkforward.py — {FAMILY} × 3 σ-variants")
    print("=" * 72)

    print("\n[1/3] Features")
    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN,
                                target='log_RK')
    print(f"  n = {len(feat)}")
    cols = FAMILIES_V2[FAMILY]

    print(f"\n[2/3] Walk-forward × 3 σ-variants (train_win={TRAIN_WIN}, refit={REFIT_STEP})")
    per_day = {}
    metrics = []
    for label, fit_fn in SIGMA_MODELS:
        print(f"\n  --- {label} ---")
        t0 = time.time()
        try:
            df_v = walk_forward_dynamic(
                feat, family_cols=cols, fit_fn=fit_fn,
                sigma_model=label, train_window=TRAIN_WIN,
                refit_step=REFIT_STEP, var_alphas=VAR_ALPHAS,
                verbose=True,
            )
        except Exception as exc:
            print(f"    FAILED: {exc!r}")
            continue
        dt = time.time() - t0
        per_day[label] = df_v
        df_v.to_parquet(TBL / f'F1_perday_{FAMILY}_{label}.parquet')
        m = _aggregate(df_v, label)
        m['fit_time_s'] = round(dt, 1)
        metrics.append(m)
        print(f"  done {dt:.0f}s n={len(df_v)}  CRPS={m['mean_crps']:.4f}  "
              f"QLIKE={m['mean_qlike']:.4f}  R²_oos={m['r2_oos']:+.4f}  "
              f"σ̂_mean={m['sigma_mean']:.3f}")

    if not metrics:
        print("No variants finished. Abort.")
        return
    mtab = pd.DataFrame(metrics)
    mtab.to_csv(TBL / f'F1_metrics_{FAMILY}.csv', index=False)

    print("\n[3/3] Pairwise DM (QLIKE) + summary")
    labels = [m['variant'] for m in metrics]
    dm = pd.DataFrame(index=labels, columns=labels, dtype=float)
    dmp = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                dm.iloc[i, j] = 0.0; dmp.iloc[i, j] = 1.0
                continue
            r = diebold_mariano(per_day[a]['qlike'].values,
                                  per_day[b]['qlike'].values, h=1)
            dm.iloc[i, j] = r['DM_hln']; dmp.iloc[i, j] = r['p_value']
    print("\nDM-HLN (negative ⇒ row beats column):")
    print(dm.round(3).to_string())
    print("\nP-values:")
    print(dmp.round(4).to_string())

    md = [f"# F1 — dynamic σ on {FAMILY} with proper MC return-VaR\n",
           f"Train window = {TRAIN_WIN}, refit every {REFIT_STEP}, target = log RK_{{t+1}}",
           f"η ~ SkewT(ν=3.0, λ=0.0)  (from C1 fitted return-innovation distribution)\n",
           "## Aggregate scores\n",
           mtab[['variant', 'n_test', 'mean_crps', 'mean_qlike',
                  'rmse_logrk', 'r2_oos', 'sigma_mean', 'sigma_std', 'fit_time_s']]
            .to_markdown(index=False, floatfmt='.4f'),
           "\n## VaR coverage — log RK upper tail\n",
           mtab[['variant'] + [f'rate_logrk_{int(a*100):02d}'
                                    for a in VAR_ALPHAS] +
                  [f'p_kup_logrk_{int(a*100):02d}'
                    for a in VAR_ALPHAS]]
            .to_markdown(index=False, floatfmt='.4f'),
           "\n## VaR coverage — RETURN lower/upper tails (MC integrated)\n",
           mtab[['variant'] + [f'rate_return_{int(a*100):02d}'
                                    for a in VAR_ALPHAS] +
                  [f'p_kup_return_{int(a*100):02d}'
                    for a in VAR_ALPHAS]]
            .to_markdown(index=False, floatfmt='.4f'),
           "\n## DM-HLN on QLIKE\n",
           dm.round(3).to_markdown(),
           "\n\nP-values:\n",
           dmp.round(4).to_markdown(),
           ]
    SUM.write_text("\n".join(md))
    print(f"\nSaved:")
    print(f"  → {SUM.relative_to(PROJECT)}")
    print(f"  → {(TBL / f'F1_metrics_{FAMILY}.csv').relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
