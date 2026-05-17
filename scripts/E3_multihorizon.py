"""E3 — Multi-horizon density forecasts via MC simulation.

Top-2 models from E1: HAR-RS-DOW (winner) and HAR-WE (sparse benchmark).
Horizons: h ∈ {1, 5, 22}.
Method: K=5000 Monte Carlo paths per test day, vectorised over paths.

For each test day t and horizon h:
    - Simulate K paths of (log RK_{t+1}, ..., log RK_{t+h})
    - Simulate corresponding return paths r_{t+1}, ..., r_{t+h}
    - Score:
        CRPS_h            on log RK_{t+h}                (h-step ahead spot)
        CRPS_avg_h        on (1/h) Σ log RK_{t+1..t+h}   (cumulative-style vol)
        QLIKE_h           on log RK_{t+h}
        VaR_α (cum r)     coverage of h-day cumulative return at α ∈ {1, 5, 95, 99}%

Output:
    outputs/tables/E3_multihorizon_perday_{family}.parquet
    outputs/tables/E3_multihorizon_summary.csv
    outputs/E3_multihorizon_summary.md
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.features_v2 import build_features_v2, FAMILIES_V2
from hartrace.estimation import fit_skewt_static
from hartrace.backtest import kupiec_test, christoffersen_test, diebold_mariano
from hartrace import distributions as skewt


warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
RV_IN = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FX_IN = PROJECT / 'data' / 'external' / 'eurusd.parquet'
FRED_IN = PROJECT / 'data' / 'external' / 'macro_fred.parquet'
TBL = PROJECT / 'outputs' / 'tables'
SUM = PROJECT / 'outputs' / 'E3_multihorizon_summary.md'

MODELS = ['HAR-RS-DOW', 'HAR-WE']
HORIZONS = [1, 5, 22]
TRAIN_WIN = 1000
REFIT_STEP = 20
N_PATHS = 5000
VAR_ALPHAS = (0.01, 0.05, 0.95, 0.99)
ETA_NU = 3.0   # return-innovation tail-index from C1
ETA_LAM = 0.0

# Proxy adjustment for log_RS_neg, log_RS_pos in simulation (since the model
# only forecasts log_RK; we assume symmetric split RS+ ≈ RS- ≈ RV/2 and
# RV ≈ RK / 0.795 from signature-plot).
# log_RS = log_RK - log(2 * 0.795) = log_RK - 0.464
RS_OFFSET = -0.464


def log(msg):
    print(msg, flush=True)


def _features_for_step(
    log_rk_d, log_rk_w, log_rk_m, dow, keep_cols, K,
):
    """Build feature matrix (K rows) for one simulation step.

    Supports columns used in HAR-RS-DOW and HAR-WE. Unused columns
    (e.g. macro_*, eurusd_*) get 0 — these models don't reference them.
    """
    feat_dict = {
        'log_RK_D': log_rk_d,
        'log_RK_W': log_rk_w,
        'log_RK_M': log_rk_m,
        'log_C_D': log_rk_d,
        'log_C_W': log_rk_w,
        'log_C_M': log_rk_m,
        'log_J_D': np.zeros(K),
        'log_J_W': np.zeros(K),
        'log_J_M': np.zeros(K),
        'log_RS_neg_D': log_rk_d + RS_OFFSET,
        'log_RS_pos_D': log_rk_d + RS_OFFSET,
        'log_RK_D_Q': np.zeros(K),
        'log_RS_neg_D_Q': np.zeros(K),
        'log_RS_pos_D_Q': np.zeros(K),
        'is_weekend': np.full(K, int(dow >= 5), dtype=float),
        'dow_1': np.full(K, int(dow == 1), dtype=float),
        'dow_2': np.full(K, int(dow == 2), dtype=float),
        'dow_3': np.full(K, int(dow == 3), dtype=float),
        'dow_4': np.full(K, int(dow == 4), dtype=float),
        'dow_5': np.full(K, int(dow == 5), dtype=float),
        'dow_6': np.full(K, int(dow == 6), dtype=float),
        'eurusd_logret': np.zeros(K),
        'eurusd_vol5': np.zeros(K),
        'macro_VIXCLS': np.zeros(K),
        'macro_T10Y2Y': np.zeros(K),
        'macro_DFF': np.zeros(K),
        'macro_DGS10': np.zeros(K),
        'macro_DGS2': np.zeros(K),
        'macro_DTWEXBGS': np.zeros(K),
    }
    return np.column_stack(
        [np.ones(K)] + [feat_dict[c] for c in keep_cols]
    )


def walk_forward_multihorizon(
    feat, family_cols, horizons, train_window, refit_step,
    n_paths, eta_nu, eta_lam, rng_seed=42,
):
    """Multi-horizon density forecasts via MC simulation."""
    rng = np.random.default_rng(rng_seed)
    keep = [c for c in family_cols if c != 'const']
    cols_needed = list(set(keep + ['target', 'r_d', 'log_RK']))
    df = feat[cols_needed].dropna().copy()
    n = len(df)
    h_max = max(horizons)
    if n < train_window + h_max + 10:
        raise RuntimeError(f"Not enough data: n={n}")

    fit_state = None
    rows = []

    for t in range(train_window, n - h_max):
        if (t - train_window) % refit_step == 0:
            sl = df.iloc[t - train_window:t]
            X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
            y_tr = sl['target'].values.astype(float)
            fit_state = fit_skewt_static(X_tr, y_tr, family_cols)
            ridx = (t - train_window) // refit_step
            if ridx % 20 == 0:
                log(f"   refit {ridx:3d} @ t={t:4d}  "
                    f"σ̂={fit_state['sigma']:.4f}, ν̂={fit_state['nu']:.2f}")

        # Build initial log_RK history of length 22
        h0 = max(0, t - 21)
        hist = df.iloc[h0:t + 1]['log_RK'].values
        if len(hist) < 22:
            hist = np.concatenate(
                [np.full(22 - len(hist), float(hist.mean())), hist])
        hist = hist.astype(float)
        history = np.tile(hist.reshape(1, 22), (n_paths, 1))

        log_rk_paths = np.empty((n_paths, h_max))
        ret_paths = np.empty((n_paths, h_max))
        date_t = df.index[t]

        for h_step in range(h_max):
            future_date = date_t + pd.Timedelta(days=h_step + 1)
            dow = future_date.dayofweek
            log_rk_d = history[:, -1]
            log_rk_w = history[:, -5:].mean(axis=1)
            log_rk_m = history[:, -22:].mean(axis=1)
            X = _features_for_step(log_rk_d, log_rk_w, log_rk_m,
                                     dow, keep, n_paths)
            mu = X @ fit_state['beta']
            eps = skewt.rvs(nu=fit_state['nu'], lam=fit_state['lam'],
                              size=n_paths, rng=rng)
            log_rk_new = mu + fit_state['sigma'] * eps
            log_rk_paths[:, h_step] = log_rk_new
            eta = skewt.rvs(nu=eta_nu, lam=eta_lam, size=n_paths, rng=rng)
            ret_paths[:, h_step] = np.exp(log_rk_new / 2.0) * eta
            history = np.concatenate(
                [history[:, 1:], log_rk_new[:, None]], axis=1)

        row = {'date': date_t}
        for h in horizons:
            # Spot at horizon end
            sim_h = log_rk_paths[:, h - 1]
            y_actual_h = float(df.iloc[t + h - 1]['target'])  # log_RK_{t+h}
            mean_pred = float(sim_h.mean())
            perm = rng.permutation(n_paths)
            crps = float(np.mean(np.abs(sim_h - y_actual_h))
                          - 0.5 * np.mean(np.abs(sim_h - sim_h[perm])))
            d = y_actual_h - mean_pred
            qlike = float(np.exp(d) - d - 1.0)
            row[f'mu_pred_h{h}'] = mean_pred
            row[f'y_actual_h{h}'] = y_actual_h
            row[f'crps_h{h}'] = crps
            row[f'qlike_h{h}'] = qlike

            # h-day average log_RK
            sim_avg = log_rk_paths[:, :h].mean(axis=1)
            actual_avg = float(
                np.mean([df.iloc[t + i]['target'] for i in range(h)]))
            perm2 = rng.permutation(n_paths)
            crps_avg = float(np.mean(np.abs(sim_avg - actual_avg))
                              - 0.5 * np.mean(np.abs(sim_avg - sim_avg[perm2])))
            row[f'crps_avg_h{h}'] = crps_avg

            # h-day cumulative return VaR
            sim_cum = ret_paths[:, :h].sum(axis=1)
            if t + h < n:
                actual_cum = float(np.sum(
                    [df.iloc[t + 1 + i]['r_d'] for i in range(h)]))
                row[f'actual_cumret_h{h}'] = actual_cum
                for a in VAR_ALPHAS:
                    pct = int(a * 100)
                    q = float(np.quantile(sim_cum, a))
                    row[f'var_cumret_h{h}_a{pct:02d}'] = q
                    viol = (int(actual_cum > q) if a >= 0.5
                             else int(actual_cum < q))
                    row[f'viol_cumret_h{h}_a{pct:02d}'] = viol
        rows.append(row)

    return pd.DataFrame(rows).set_index('date')


def summarize(per_day: pd.DataFrame, family: str) -> dict:
    out = {'family': family, 'n': len(per_day)}
    for h in HORIZONS:
        out[f'crps_h{h}'] = float(per_day[f'crps_h{h}'].mean())
        out[f'crps_avg_h{h}'] = float(per_day[f'crps_avg_h{h}'].mean())
        out[f'qlike_h{h}'] = float(per_day[f'qlike_h{h}'].mean())
        out[f'rmse_h{h}'] = float(np.sqrt(
            ((per_day[f'y_actual_h{h}'] - per_day[f'mu_pred_h{h}']) ** 2).mean()))
        for a in VAR_ALPHAS:
            pct = int(a * 100)
            target = a if a < 0.5 else 1 - a
            col = f'viol_cumret_h{h}_a{pct:02d}'
            if col in per_day.columns:
                v = per_day[col].dropna().values.astype(int)
                if len(v) > 0:
                    out[f'rate_cumret_h{h}_a{pct:02d}'] = float(v.mean())
                    out[f'kup_p_cumret_h{h}_a{pct:02d}'] = kupiec_test(v, target)['p_value']
    return out


def main():
    log("=" * 72)
    log("E3 — Multi-horizon density forecasts via MC simulation")
    log("=" * 72)

    feat = build_features_v2(RV_IN, eurusd_path=FX_IN, fred_path=FRED_IN,
                              target='log_RK')
    log(f"feat n={len(feat)}")
    log(f"models: {MODELS}, horizons: {HORIZONS}, K_paths={N_PATHS}")

    summaries = []
    per_day_by_fam = {}
    for fam in MODELS:
        log(f"\n--- {fam} ---")
        t0 = time.time()
        cols = FAMILIES_V2[fam]
        df_fam = walk_forward_multihorizon(
            feat, family_cols=cols, horizons=HORIZONS,
            train_window=TRAIN_WIN, refit_step=REFIT_STEP,
            n_paths=N_PATHS, eta_nu=ETA_NU, eta_lam=ETA_LAM,
        )
        df_fam.to_parquet(TBL / f'E3_multihorizon_perday_{fam}.parquet')
        s = summarize(df_fam, fam)
        s['fit_s'] = round(time.time() - t0, 1)
        summaries.append(s)
        per_day_by_fam[fam] = df_fam
        log(f"  {fam} done {time.time() - t0:.1f}s, n={len(df_fam)}")
        log(f"  CRPS:   h=1 {s['crps_h1']:.4f},  h=5 {s['crps_h5']:.4f},  "
            f"h=22 {s['crps_h22']:.4f}")
        log(f"  CRPS_avg: h=5 {s['crps_avg_h5']:.4f},  h=22 {s['crps_avg_h22']:.4f}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(TBL / 'E3_multihorizon_summary.csv', index=False)

    log("\n=== SUMMARY ===")
    cols_crps = ['family'] + [f'crps_h{h}' for h in HORIZONS] \
                + [f'crps_avg_h{h}' for h in HORIZONS if h > 1]
    log(summary[cols_crps].to_string(index=False))

    log("\nCumulative-return VaR coverage by horizon (target = α for tails 0.01/0.05):")
    for h in HORIZONS:
        rate_cols = [f'rate_cumret_h{h}_a{int(a*100):02d}' for a in VAR_ALPHAS]
        kup_cols = [f'kup_p_cumret_h{h}_a{int(a*100):02d}' for a in VAR_ALPHAS]
        log(f"\n  h={h}:")
        log(summary[['family'] + rate_cols].to_string(index=False))
        log(summary[['family'] + kup_cols].to_string(index=False))

    log("\nDM (CRPS h=1, then h=5, h=22) — row vs column, negative ⇒ row wins")
    for h in HORIZONS:
        log(f"\n  h={h}")
        a = per_day_by_fam[MODELS[0]][f'crps_h{h}'].values
        b = per_day_by_fam[MODELS[1]][f'crps_h{h}'].values
        dm = diebold_mariano(a, b, h=h)
        log(f"    {MODELS[0]} vs {MODELS[1]}: DM_hln={dm['DM_hln']:+.3f}, "
            f"p={dm['p_value']:.4f}, mean diff={dm['mean_diff']:+.5f}")

    # Markdown
    md = ["# E3 — Multi-horizon density forecasts (h=1, 5, 22)\n",
           f"K={N_PATHS} MC paths per test day. train_win={TRAIN_WIN}, "
           f"refit={REFIT_STEP}.\n",
           "## CRPS by horizon\n",
           summary[cols_crps].to_markdown(index=False, floatfmt='.4f'),
           "\n## Cum-return VaR Kupiec rates\n",
           ]
    for h in HORIZONS:
        rate_cols = [f'rate_cumret_h{h}_a{int(a*100):02d}' for a in VAR_ALPHAS]
        kup_cols = [f'kup_p_cumret_h{h}_a{int(a*100):02d}' for a in VAR_ALPHAS]
        md.append(f"\n### h={h}\n")
        md.append("Rates:\n")
        md.append(summary[['family'] + rate_cols].to_markdown(index=False, floatfmt='.4f'))
        md.append("Kupiec p:\n")
        md.append(summary[['family'] + kup_cols].to_markdown(index=False, floatfmt='.4f'))
    SUM.write_text("\n".join(md))
    log(f"\nSaved: {SUM}")
    log("\nDone.\n")


if __name__ == '__main__':
    main()
