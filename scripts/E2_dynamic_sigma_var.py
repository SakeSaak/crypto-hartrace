"""Pad B — dynamic σ + MC-integrated return-VaR on HAR-RS-DOW.

Tests three σ specifications × two VaR methods on the winning model:

  σ specifications:
    static  σ                          (constant, from skew-t MLE)
    GAS     σ_t                        (score-driven; Creal et al. 2013)
    EGARCH  σ_t                        (asymmetric exponential GARCH)

  VaR methods (return-VaR specifically):
    plug-in   σ̂_ret = exp(μ̂_logRK / 2), VaR_α = σ̂_ret · F_η^{-1}(α)
    MC-int    sample (ε, η), build r* = exp((μ̂ + σ̂·ε)/2) · η, take α-quantile

Pre-registered hypothesis (after E1):
  - Plug-in return-VaR is under-covered (~10% violations at 5% target)
  - MC integration alone may fix this (accounts for vol-of-vol)
  - Dynamic σ may further improve density-forecast quality
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
    walk_forward_h1, walk_forward_h1_dynamic,
    kupiec_test, christoffersen_test, diebold_mariano,
)

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E2_dynamic_sigma_summary.md'

FAMILY = 'HAR-RS-DOW'
TRAIN_WIN = 1000
REFIT_STEP = 20
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)


def _wf_static(feat, cols):
    return walk_forward_h1(
        feat, family_cols=cols, fit_fn=fit_skewt_static,
        train_window=TRAIN_WIN, refit_step=REFIT_STEP,
        var_alphas=VAR_ALPHAS, verbose=True, n_mc_crps=2000,
    )


def _wf_gas(feat, cols):
    return walk_forward_h1_dynamic(
        feat, family_cols=cols, fit_fn=fit_skewt_gas,
        scale_type='gas',
        train_window=TRAIN_WIN, refit_step=REFIT_STEP,
        var_alphas=VAR_ALPHAS, verbose=True,
        n_mc_crps=2000, n_mc_var=4000,
    )


def _wf_egarch(feat, cols):
    return walk_forward_h1_dynamic(
        feat, family_cols=cols, fit_fn=fit_skewt_egarch,
        scale_type='egarch',
        train_window=TRAIN_WIN, refit_step=REFIT_STEP,
        var_alphas=VAR_ALPHAS, verbose=True,
        n_mc_crps=2000, n_mc_var=4000,
    )


def _post_hoc_mc_var(per_day: pd.DataFrame, eta_nu=3.0, eta_lam=0.0,
                      n_mc=4000, rng_seed=42) -> pd.DataFrame:
    """For each row, compute MC-integrated return-VaR using mu/sigma/nu/lam."""
    from hartrace import distributions as skewt
    rng = np.random.default_rng(rng_seed)
    out = per_day.copy()
    for a in VAR_ALPHAS:
        out[f'var_return_mc_{int(a*100):02d}'] = np.nan
        out[f'viol_return_mc_{int(a*100):02d}'] = 0
    for i, row in enumerate(per_day.itertuples()):
        eps = skewt.rvs(nu=row.nu_pred, lam=row.lam_pred, size=n_mc, rng=rng)
        eta = skewt.rvs(nu=eta_nu, lam=eta_lam, size=n_mc, rng=rng)
        logrk_sim = row.mu_pred + row.sigma_pred * eps
        r_sim = np.exp(logrk_sim / 2.0) * eta
        for a in VAR_ALPHAS:
            q = float(np.quantile(r_sim, a))
            out.iloc[i, out.columns.get_loc(f'var_return_mc_{int(a*100):02d}')] = q
            viol = int(row.r_actual > q) if a >= 0.5 else int(row.r_actual < q)
            out.iloc[i, out.columns.get_loc(f'viol_return_mc_{int(a*100):02d}')] = viol
    return out


def _summarize(per_day: pd.DataFrame, label: str) -> dict:
    out = {'spec': label, 'n': len(per_day),
           'mean_crps': float(per_day['crps'].mean()),
           'mean_qlike': float(per_day['qlike'].mean()),
           'rmse_logrk': float(np.sqrt(((per_day['y_actual'] - per_day['mu_pred']) ** 2).mean())),
           'r2_oos': float(1 - ((per_day['y_actual'] - per_day['mu_pred']) ** 2).sum()
                            / ((per_day['y_actual'] - per_day['y_actual'].mean()) ** 2).sum())}
    for a in VAR_ALPHAS:
        a_test = a if a < 0.5 else 1 - a
        # logrk
        v = per_day[f'viol_logrk_{int(a*100):02d}'].values
        out[f'logrk_rate_{int(a*100):02d}'] = float(v.mean())
        out[f'logrk_kupiec_p_{int(a*100):02d}'] = kupiec_test(v, a_test)['p_value']
        # return plug
        if f'var_return_plug_{int(a*100):02d}' in per_day.columns:
            vr = per_day[f'viol_return_plug_{int(a*100):02d}'].values
            out[f'rplug_rate_{int(a*100):02d}'] = float(vr.mean())
            out[f'rplug_kupiec_p_{int(a*100):02d}'] = kupiec_test(vr, a_test)['p_value']
        # return MC
        if f'viol_return_mc_{int(a*100):02d}' in per_day.columns:
            vmc = per_day[f'viol_return_mc_{int(a*100):02d}'].values
            out[f'rmc_rate_{int(a*100):02d}'] = float(vmc.mean())
            out[f'rmc_kupiec_p_{int(a*100):02d}'] = kupiec_test(vmc, a_test)['p_value']
    return out


def main():
    print("=" * 72)
    print(f"E2 — dynamic σ + MC-VaR on {FAMILY}")
    print("=" * 72)

    print("\n[1/4] Features")
    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN,
                                target='log_RK')
    cols = FAMILIES_V2[FAMILY]
    print(f"  family columns ({len(cols)}): {cols}")

    print("\n[2/4] Walk-forward static σ (recomputed with MC-VaR post-hoc)")
    t0 = time.time()
    pd_static = _wf_static(feat, cols)
    pd_static = _post_hoc_mc_var(pd_static)
    pd_static.to_parquet(TBL / f'E2_static_perday_{FAMILY}.parquet')
    print(f"  done {time.time()-t0:.1f}s, n_test={len(pd_static)}")

    print("\n[3/4] Walk-forward GAS σ_t")
    t0 = time.time()
    try:
        pd_gas = _wf_gas(feat, cols)
        pd_gas.to_parquet(TBL / f'E2_gas_perday_{FAMILY}.parquet')
        print(f"  done {time.time()-t0:.1f}s")
        gas_ok = True
    except Exception as exc:
        print(f"  GAS failed: {exc!r}")
        pd_gas = None
        gas_ok = False

    print("\n[4/4] Walk-forward EGARCH σ_t")
    t0 = time.time()
    try:
        pd_egarch = _wf_egarch(feat, cols)
        pd_egarch.to_parquet(TBL / f'E2_egarch_perday_{FAMILY}.parquet')
        print(f"  done {time.time()-t0:.1f}s")
        egarch_ok = True
    except Exception as exc:
        print(f"  EGARCH failed: {exc!r}")
        pd_egarch = None
        egarch_ok = False

    print("\n=== SUMMARY ===")
    rows = [_summarize(pd_static, 'static')]
    if gas_ok:    rows.append(_summarize(pd_gas, 'gas'))
    if egarch_ok: rows.append(_summarize(pd_egarch, 'egarch'))
    summ = pd.DataFrame(rows)
    summ.to_csv(TBL / 'E2_summary.csv', index=False)

    print("\n## Density quality")
    print(summ[['spec', 'mean_crps', 'mean_qlike', 'rmse_logrk', 'r2_oos']].to_string(index=False))

    print("\n## log-RK VaR (Kupiec rates — target=α)")
    log_cols = [f'{p}{int(a*100):02d}' for p in ('logrk_rate_',)
                  for a in VAR_ALPHAS]
    print(summ[['spec'] + log_cols].to_string(index=False))

    print("\n## Return-VaR plug-in (target left-tail 5% and 1%)")
    pcols = ['rplug_rate_05', 'rplug_kupiec_p_05',
             'rplug_rate_01', 'rplug_kupiec_p_01']
    print(summ[['spec'] + pcols].to_string(index=False))

    print("\n## Return-VaR MC-INTEGRATED (target left-tail 5% and 1%)")
    mcols = ['rmc_rate_05', 'rmc_kupiec_p_05',
             'rmc_rate_01', 'rmc_kupiec_p_01']
    print(summ[['spec'] + mcols].to_string(index=False))

    # DM comparisons on QLIKE between specs
    print("\n## Diebold-Mariano (QLIKE) — row vs column, negative ⇒ row wins")
    specs = [r['spec'] for r in rows]
    dms = pd.DataFrame(index=specs, columns=specs, dtype=float)
    pvs = pd.DataFrame(index=specs, columns=specs, dtype=float)
    by_spec = {'static': pd_static}
    if gas_ok: by_spec['gas'] = pd_gas
    if egarch_ok: by_spec['egarch'] = pd_egarch
    for sa in specs:
        for sb in specs:
            if sa == sb:
                dms.loc[sa, sb] = 0.0; pvs.loc[sa, sb] = 1.0; continue
            d = diebold_mariano(by_spec[sa]['qlike'].values,
                                  by_spec[sb]['qlike'].values, h=1)
            dms.loc[sa, sb] = d['DM_hln']; pvs.loc[sa, sb] = d['p_value']
    print(dms.round(3).to_string())
    print("\nP-values:")
    print(pvs.round(4).to_string())
    dms.to_csv(TBL / 'E2_dm_qlike.csv')
    pvs.to_csv(TBL / 'E2_dm_pvalues.csv')

    # Markdown summary
    md = [f"# E2 — Dynamic σ + MC-VaR on {FAMILY}\n",
           f"Walk-forward h=1, train_win={TRAIN_WIN}, refit={REFIT_STEP}.\n",
           "## Density quality\n",
           summ[['spec', 'mean_crps', 'mean_qlike', 'rmse_logrk',
                  'r2_oos']].to_markdown(index=False, floatfmt='.4f'),
           "\n## Return-VaR plug-in vs MC-integrated\n",
           summ[['spec'] + pcols + mcols].to_markdown(index=False, floatfmt='.4f'),
           "\n## DM-HLN (QLIKE)\n",
           dms.round(3).to_markdown(),
           "\nP-values:\n",
           pvs.round(4).to_markdown(),
           ]
    SUM.write_text("\n".join(md))
    print(f"\nSaved: {SUM.relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
