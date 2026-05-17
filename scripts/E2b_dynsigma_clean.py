"""E2b — clean rerun GAS + EGARCH walk-forward on HAR-RS-DOW.

Differs from E2:
    - verbose=False (no print spam → no BrokenPipeError when stdout breaks)
    - smaller n_mc (1500/3000) for faster turnaround
    - skips static (already in E2_static_perday_HAR-RS-DOW.parquet)
    - direct file output, no `tee` middleman

Run with:
    cd ~/Desktop/crypto_hartrace
    nohup /opt/anaconda3/bin/python -u scripts/E2b_dynsigma_clean.py > /tmp/E2b.log 2>&1 &
    echo $!   # remember the PID for monitoring
"""

from __future__ import annotations
import sys, time, warnings, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_gas, fit_skewt_egarch
from hartrace.backtest import walk_forward_h1_dynamic, kupiec_test


warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E2b_dynamic_sigma_summary.md'

FAMILY = 'HAR-RS-DOW'
TRAIN_WIN = 1000
REFIT_STEP = 20
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 60)
    log(f"E2b — GAS + EGARCH on {FAMILY} (clean rerun)")
    log("=" * 60)

    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN,
                              target='log_RK')
    cols = FAMILIES_V2[FAMILY]
    log(f"feat n={len(feat)}, family cols={len(cols)}")

    # ===== GAS =====
    log("\n[1/2] GAS walk-forward")
    t0 = time.time()
    pd_gas = None
    gas_ok = False
    try:
        pd_gas = walk_forward_h1_dynamic(
            feat, family_cols=cols, fit_fn=fit_skewt_gas,
            scale_type='gas',
            train_window=TRAIN_WIN, refit_step=REFIT_STEP,
            var_alphas=VAR_ALPHAS, verbose=False,
            n_mc_crps=1500, n_mc_var=3000,
        )
        pd_gas.to_parquet(TBL / f'E2b_gas_perday_{FAMILY}.parquet')
        log(f"  GAS done: {time.time() - t0:.1f}s, n={len(pd_gas)}")
        gas_ok = True
    except Exception:
        log("  GAS FAILED with traceback:")
        log(traceback.format_exc())

    # ===== EGARCH =====
    log("\n[2/2] EGARCH walk-forward")
    t0 = time.time()
    pd_egarch = None
    egarch_ok = False
    try:
        pd_egarch = walk_forward_h1_dynamic(
            feat, family_cols=cols, fit_fn=fit_skewt_egarch,
            scale_type='egarch',
            train_window=TRAIN_WIN, refit_step=REFIT_STEP,
            var_alphas=VAR_ALPHAS, verbose=False,
            n_mc_crps=1500, n_mc_var=3000,
        )
        pd_egarch.to_parquet(TBL / f'E2b_egarch_perday_{FAMILY}.parquet')
        log(f"  EGARCH done: {time.time() - t0:.1f}s, n={len(pd_egarch)}")
        egarch_ok = True
    except Exception:
        log("  EGARCH FAILED with traceback:")
        log(traceback.format_exc())

    # ===== Summary against static baseline =====
    log("\n=== Summary ===")
    rows = []
    static_path = TBL / f'E2_static_perday_{FAMILY}.parquet'
    if static_path.exists():
        ps = pd.read_parquet(static_path)
        rows.append(('static', ps))
    if gas_ok:
        rows.append(('gas', pd_gas))
    if egarch_ok:
        rows.append(('egarch', pd_egarch))

    summary = []
    for label, df in rows:
        r = {'spec': label,
              'mean_crps': float(df['crps'].mean()),
              'mean_qlike': float(df['qlike'].mean()),
              'rmse_logrk': float(np.sqrt(((df['y_actual'] - df['mu_pred']) ** 2).mean())),
              'r2_oos': float(1 - ((df['y_actual'] - df['mu_pred']) ** 2).sum()
                              / ((df['y_actual'] - df['y_actual'].mean()) ** 2).sum())}
        for a in VAR_ALPHAS:
            pct = int(a * 100)
            target = a if a < 0.5 else 1 - a
            col_plug = f'viol_return_{pct:02d}' if f'viol_return_{pct:02d}' in df.columns \
                        else f'viol_return_plug_{pct:02d}'
            col_mc = f'viol_return_mc_{pct:02d}'
            if col_plug in df.columns:
                r[f'rplug_{pct:02d}'] = float(df[col_plug].mean())
                r[f'rplug_kp_{pct:02d}'] = kupiec_test(df[col_plug].values, target)['p_value']
            if col_mc in df.columns:
                r[f'rmc_{pct:02d}'] = float(df[col_mc].mean())
                r[f'rmc_kp_{pct:02d}'] = kupiec_test(df[col_mc].values, target)['p_value']
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary.to_csv(TBL / 'E2b_summary.csv', index=False)

    log("\nDensity quality:")
    log(summary[['spec', 'mean_crps', 'mean_qlike', 'rmse_logrk', 'r2_oos']]
        .to_string(index=False))
    if 'rmc_05' in summary.columns:
        log("\nReturn-VaR MC-int rates:")
        cols2 = ['spec'] + [f'rmc_{int(a*100):02d}' for a in VAR_ALPHAS]
        log(summary[cols2].to_string(index=False))

    md = [f"# E2b — Dynamic σ rerun (clean) on {FAMILY}\n",
           "## Density quality\n",
           summary.to_markdown(index=False, floatfmt='.4f')]
    SUM.write_text("\n".join(md))
    log(f"\nSaved: {SUM}")
    log("Done.")


if __name__ == '__main__':
    main()
