"""E4b — gecorrigeerde versie van E4.

Fixes:
  - h_step=0 gebruikt OBSERVED features van rij t (incl. werkelijke RS+/RS-,
    correcte dow voor row t).
  - h_step > 0 gebruikt dow van (date_t + h_step) als PREDICTOR, en alleen
    voor semi-variances een proxy omdat we die niet expliciet modelleren.
"""
from __future__ import annotations
import sys, time, warnings, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np, pandas as pd
from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_static
from hartrace.backtest import kupiec_test, diebold_mariano
from hartrace import distributions as skewt
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E4b_combined_summary.md'

MODELS_A = ['HAR-RS-DOW', 'HAR-WE']
HORIZONS = [1, 5]
TRAIN_WIN, REFIT_STEP, N_PATHS = 1000, 20, 2500
SAVE_EVERY = 100
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)
ETA_NU, ETA_LAM = 3.0, 0.0
RS_OFFSET = -0.464

def log(m): print(m, flush=True)

def _feat_step_simulated(log_rk_d, log_rk_w, log_rk_m, dow, keep, K):
    """Features for SIMULATED day (h_step > 0). Uses log_RK history + proxy for RS."""
    d = {'log_RK_D': log_rk_d, 'log_RK_W': log_rk_w, 'log_RK_M': log_rk_m,
        'log_RS_neg_D': log_rk_d+RS_OFFSET, 'log_RS_pos_D': log_rk_d+RS_OFFSET,
        'is_weekend': np.full(K, int(dow>=5), dtype=float)}
    for i in range(1, 7):
        d[f'dow_{i}'] = np.full(K, int(dow==i), dtype=float)
    for c in ('log_C_D','log_C_W','log_C_M','log_J_D','log_J_W','log_J_M',
              'log_RK_D_Q','log_RS_neg_D_Q','log_RS_pos_D_Q','eurusd_logret',
              'eurusd_vol5','macro_VIXCLS','macro_T10Y2Y','macro_DFF',
              'macro_DGS10','macro_DGS2','macro_DTWEXBGS'):
        if c not in d:
            d[c] = log_rk_d if c.startswith('log_C') else np.zeros(K)
    return np.column_stack([np.ones(K)] + [d[c] for c in keep])

def wf_multihorizon(feat, family_cols, family_name):
    rng = np.random.default_rng(42)
    keep = [c for c in family_cols if c != 'const']
    cols_needed = list(set(keep + ['target', 'r_d', 'log_RK']))
    df = feat[cols_needed].dropna().copy()
    n = len(df); h_max = max(HORIZONS)
    fit_state = None; rows = []
    n_iter = 0
    save_path = TBL / f'E4b_A_perday_{family_name}.parquet'
    for t in range(TRAIN_WIN, n - h_max):
        if (t - TRAIN_WIN) % REFIT_STEP == 0:
            sl = df.iloc[t-TRAIN_WIN:t]
            X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
            y_tr = sl['target'].values.astype(float)
            fit_state = fit_skewt_static(X_tr, y_tr, family_cols)
            ridx = (t - TRAIN_WIN) // REFIT_STEP
            if ridx % 10 == 0:
                log(f"   {family_name} refit {ridx} @ t={t}  σ̂={fit_state['sigma']:.3f}, ν̂={fit_state['nu']:.1f}")
        h0 = max(0, t - 21)
        hist = df.iloc[h0:t+1]['log_RK'].values.astype(float)
        if len(hist) < 22:
            hist = np.concatenate([np.full(22-len(hist), float(hist.mean())), hist])
        history = np.tile(hist.reshape(1, 22), (N_PATHS, 1))
        lp = np.empty((N_PATHS, h_max)); rp = np.empty((N_PATHS, h_max))
        date_t = df.index[t]
        for h_step in range(h_max):
            if h_step == 0:
                # FIX: use OBSERVED features from row t (correct dow + true RS+/RS-)
                x_obs = df.iloc[t][keep].values.astype(float)
                X = np.tile(np.concatenate([[1.0], x_obs])[None, :], (N_PATHS, 1))
            else:
                # FIX: predictor_date = date_t + h_step (one before target)
                predictor_date = date_t + pd.Timedelta(days=h_step)
                dow = predictor_date.dayofweek
                lrd = history[:, -1]
                lrw = history[:, -5:].mean(axis=1)
                lrm = history[:, -22:].mean(axis=1)
                X = _feat_step_simulated(lrd, lrw, lrm, dow, keep, N_PATHS)
            mu = X @ fit_state['beta']
            eps = skewt.rvs(nu=fit_state['nu'], lam=fit_state['lam'], size=N_PATHS, rng=rng)
            lp[:, h_step] = mu + fit_state['sigma'] * eps
            eta = skewt.rvs(nu=ETA_NU, lam=ETA_LAM, size=N_PATHS, rng=rng)
            rp[:, h_step] = np.exp(lp[:, h_step] / 2.0) * eta
            history = np.concatenate([history[:, 1:], lp[:, h_step:h_step+1]], axis=1)
        row = {'date': date_t}
        for h in HORIZONS:
            sim_h = lp[:, h-1]; ya = float(df.iloc[t+h-1]['target'])
            mp = float(sim_h.mean()); perm = rng.permutation(N_PATHS)
            crps = float(np.mean(np.abs(sim_h-ya)) - 0.5*np.mean(np.abs(sim_h-sim_h[perm])))
            d_ = ya - mp
            row[f'mu_h{h}'] = mp; row[f'y_h{h}'] = ya
            row[f'crps_h{h}'] = crps; row[f'qlike_h{h}'] = float(np.exp(d_)-d_-1.0)
            sim_cum = rp[:, :h].sum(axis=1)
            if t+h < n:
                ac = float(np.sum([df.iloc[t+1+i]['r_d'] for i in range(h)]))
                row[f'actual_cum_h{h}'] = ac
                for a in VAR_ALPHAS:
                    pct = int(a*100); q = float(np.quantile(sim_cum, a))
                    row[f'var_cum_h{h}_a{pct:02d}'] = q
                    row[f'viol_cum_h{h}_a{pct:02d}'] = int(ac > q) if a >= 0.5 else int(ac < q)
        rows.append(row); n_iter += 1
        if n_iter % SAVE_EVERY == 0:
            pd.DataFrame(rows).set_index('date').to_parquet(save_path)
    pd.DataFrame(rows).set_index('date').to_parquet(save_path)
    return pd.DataFrame(rows).set_index('date')

def main():
    log("E4b — bugfix multi-horizon (Pad A only first)")
    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN, target='log_RK')
    log(f"feat n={len(feat)}")
    results = {}
    for fam in MODELS_A:
        log(f"\n--- {fam} ---")
        t0 = time.time()
        results[fam] = wf_multihorizon(feat, FAMILIES_V2[fam], fam)
        log(f"  {fam} done {time.time()-t0:.1f}s, n={len(results[fam])}")
        log(f"  CRPS_h1 = {results[fam]['crps_h1'].mean():.4f}, "
            f"CRPS_h5 = {results[fam]['crps_h5'].mean():.4f}")
    # Summary
    log("\n=== SUMMARY ===")
    print(f"{'family':<14}{'n':>6}{'CRPS_h1':>11}{'QLIKE_h1':>11}{'R2_h1':>11}{'CRPS_h5':>11}{'R2_h5':>11}")
    for fam, df in results.items():
        r2_1 = 1 - ((df['y_h1']-df['mu_h1'])**2).sum() / ((df['y_h1']-df['y_h1'].mean())**2).sum()
        r2_5 = 1 - ((df['y_h5']-df['mu_h5'])**2).sum() / ((df['y_h5']-df['y_h5'].mean())**2).sum()
        print(f"{fam:<14}{len(df):>6}{df['crps_h1'].mean():>11.4f}"
              f"{df['qlike_h1'].mean():>11.4f}{r2_1:>11.4f}"
              f"{df['crps_h5'].mean():>11.4f}{r2_5:>11.4f}")
    log("\nVaR coverage at h=1 (target = α for 0.01,0.05; 1-α for 0.95,0.99):")
    for fam, df in results.items():
        log(f"  {fam}:")
        for a in VAR_ALPHAS:
            pct = int(a*100); target = a if a<0.5 else 1-a
            v = df[f'viol_cum_h1_a{pct:02d}'].dropna().astype(int).values
            p = kupiec_test(v, target)['p_value']
            log(f"    α={a:.2f}  rate={v.mean():.4f}  target={target:.4f}  Kup-p={p:.4f}")
    log("\nDM (CRPS): HAR-RS-DOW vs HAR-WE")
    for h in HORIZONS:
        dm = diebold_mariano(results['HAR-RS-DOW'][f'crps_h{h}'].values,
                              results['HAR-WE'][f'crps_h{h}'].values, h=h)
        log(f"  h={h}: DM={dm['DM_hln']:+.3f}, p={dm['p_value']:.4f}, "
            f"meandiff={dm['mean_diff']:+.5f}")
    log("\nDone.")

if __name__ == '__main__': main()
