"""E4 — Combined leaner Pad A + Pad B.

Pad A: HAR-RS-DOW + HAR-WE at h=[1, 5], K=2500 paths.
Pad B: GAS on HAR-RS-DOW only.
Incremental save every 100 test days → safe against abort.
Runtime ~10 min sequentially in foreground.
"""
from __future__ import annotations
import sys, time, warnings, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np, pandas as pd
from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_static, fit_skewt_gas
from hartrace.backtest import walk_forward_h1_dynamic, kupiec_test, diebold_mariano
from hartrace import distributions as skewt
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E4_combined_summary.md'

MODELS_A = ['HAR-RS-DOW', 'HAR-WE']
HORIZONS = [1, 5]
TRAIN_WIN, REFIT_STEP, N_PATHS = 1000, 20, 2500
SAVE_EVERY = 100
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)
ETA_NU, ETA_LAM = 3.0, 0.0
RS_OFFSET = -0.464

def log(m): print(m, flush=True)

def _feat_step(log_rk_d, log_rk_w, log_rk_m, dow, keep, K):
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
    save_path = TBL / f'E4_A_perday_{family_name}.parquet'
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
            fd = date_t + pd.Timedelta(days=h_step+1)
            dow = fd.dayofweek
            lrd = history[:, -1]; lrw = history[:, -5:].mean(axis=1); lrm = history[:, -22:].mean(axis=1)
            X = _feat_step(lrd, lrw, lrm, dow, keep, N_PATHS)
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
            log(f"   {family_name} saved partial at t={t}, n_rows={n_iter}")
    df_out = pd.DataFrame(rows).set_index('date')
    df_out.to_parquet(save_path)
    return df_out

def main():
    log("=" * 60)
    log("E4 — Combined Pad A (h=1,5) + Pad B (GAS) leaner")
    log("=" * 60)
    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN, target='log_RK')
    log(f"feat n={len(feat)}")

    # === PAD A: Multi-horizon ===
    log("\n## PAD A: Multi-horizon density forecasts ##")
    pad_a = {}
    for fam in MODELS_A:
        log(f"\n--- {fam} ---")
        t0 = time.time()
        try:
            pad_a[fam] = wf_multihorizon(feat, FAMILIES_V2[fam], fam)
            log(f"  {fam} done {time.time()-t0:.1f}s, n={len(pad_a[fam])}")
        except Exception:
            log(traceback.format_exc())

    # === PAD B: GAS on HAR-RS-DOW ===
    log("\n## PAD B: GAS dynamic σ on HAR-RS-DOW ##")
    pad_b_gas = None
    try:
        t0 = time.time()
        cols = FAMILIES_V2['HAR-RS-DOW']
        pad_b_gas = walk_forward_h1_dynamic(
            feat, family_cols=cols, fit_fn=fit_skewt_gas,
            scale_type='gas', train_window=TRAIN_WIN, refit_step=REFIT_STEP,
            var_alphas=VAR_ALPHAS, verbose=False, n_mc_crps=1500, n_mc_var=3000)
        pad_b_gas.to_parquet(TBL / 'E4_B_gas_perday_HAR-RS-DOW.parquet')
        log(f"  GAS done {time.time()-t0:.1f}s, n={len(pad_b_gas)}")
    except Exception:
        log(traceback.format_exc())

    # === Summary ===
    log("\n=== SUMMARY ===")
    md = ["# E4 — Combined results\n"]

    # Pad A summary
    log("\n## Pad A: Multi-horizon CRPS")
    rows_a = []
    for fam, df_fam in pad_a.items():
        r = {'family': fam, 'n': len(df_fam)}
        for h in HORIZONS:
            r[f'crps_h{h}'] = float(df_fam[f'crps_h{h}'].mean())
            r[f'qlike_h{h}'] = float(df_fam[f'qlike_h{h}'].mean())
            for a in VAR_ALPHAS:
                pct, target = int(a*100), (a if a<0.5 else 1-a)
                col = f'viol_cum_h{h}_a{pct:02d}'
                if col in df_fam.columns:
                    v = df_fam[col].dropna().values.astype(int)
                    if len(v) > 0:
                        r[f'rate_cum_h{h}_a{pct:02d}'] = float(v.mean())
                        r[f'kp_cum_h{h}_a{pct:02d}'] = kupiec_test(v, target)['p_value']
        rows_a.append(r)
    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(TBL / 'E4_A_summary.csv', index=False)
    crps_cols = ['family','n'] + [f'crps_h{h}' for h in HORIZONS] + [f'qlike_h{h}' for h in HORIZONS]
    log(df_a[crps_cols].to_string(index=False))
    md.append("## Pad A — Multi-horizon density quality\n")
    md.append(df_a[crps_cols].to_markdown(index=False, floatfmt='.4f'))
    for h in HORIZONS:
        log(f"\nh={h} cum-return VaR coverage (rate vs target):")
        rc = ['family'] + [f'rate_cum_h{h}_a{int(a*100):02d}' for a in VAR_ALPHAS]
        if all(c in df_a.columns for c in rc):
            log(df_a[rc].to_string(index=False))
            md.append(f"\n### h={h} VaR rates\n")
            md.append(df_a[rc].to_markdown(index=False, floatfmt='.4f'))

    # Pad B summary
    log("\n## Pad B: GAS vs Static on HAR-RS-DOW")
    rows_b = []
    sp = TBL / 'E2_static_perday_HAR-RS-DOW.parquet'
    if sp.exists():
        rows_b.append(('static', pd.read_parquet(sp)))
    if pad_b_gas is not None:
        rows_b.append(('gas', pad_b_gas))
    summ_b = []
    for label, dfx in rows_b:
        r = {'spec': label, 'mean_crps': float(dfx['crps'].mean()),
             'mean_qlike': float(dfx['qlike'].mean())}
        for a in VAR_ALPHAS:
            pct, target = int(a*100), (a if a<0.5 else 1-a)
            cp = f'viol_return_{pct:02d}' if f'viol_return_{pct:02d}' in dfx.columns else f'viol_return_plug_{pct:02d}'
            cm = f'viol_return_mc_{pct:02d}'
            if cp in dfx.columns: r[f'rplug_{pct:02d}'] = float(dfx[cp].mean())
            if cm in dfx.columns: r[f'rmc_{pct:02d}'] = float(dfx[cm].mean())
        summ_b.append(r)
    df_b = pd.DataFrame(summ_b)
    df_b.to_csv(TBL / 'E4_B_summary.csv', index=False)
    log(df_b.to_string(index=False))
    md.append("\n## Pad B — GAS vs Static (return-VaR focus)\n")
    md.append(df_b.to_markdown(index=False, floatfmt='.4f'))

    # DM Pad A
    if len(pad_a) == 2:
        log("\n## DM tests (Pad A, CRPS by horizon)")
        f1, f2 = list(pad_a.keys())
        for h in HORIZONS:
            dm = diebold_mariano(pad_a[f1][f'crps_h{h}'].values,
                                   pad_a[f2][f'crps_h{h}'].values, h=h)
            log(f"  h={h}: {f1} vs {f2}: DM={dm['DM_hln']:+.3f}, p={dm['p_value']:.4f}")

    SUM.write_text("\n".join(md))
    log(f"\nSaved: {SUM}\nDone.")

if __name__ == '__main__': main()
