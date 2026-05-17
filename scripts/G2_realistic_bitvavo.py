"""G2 — Optimalisatie onder realistische Bitvavo-constraints.

Long-only (Bitvavo = spot-only, geen leverage/margin).
w ∈ [0, 1]. Onder hoge vol → naar cash. Lage vol → max BTC-exposure.

Grid: 3 modellen × 4 vol-targets × 3 thresholds × 3 fee-scenarios = 108 configs.
Plus baselines: buy-and-hold (passief), DCA wekelijks (€10/week).

Target: maximum risk-adjusted return (Sharpe) onder vooraf gekozen risico-tolerantie.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/Users/sakesaakstra/Desktop/crypto_hartrace/src')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT = '/Users/sakesaakstra/Desktop/crypto_hartrace'

# Bitvavo fees (base tier 2026, BTC-EUR)
FEES = {'maker_15bps': 0.0015, 'blend_20bps': 0.0020, 'taker_25bps': 0.0025}

# Long-only constraint
MAX_W = 1.0
MIN_W = 0.0


def vol_managed_lo(sigma_pred, returns, target_vol_ann, fee, rebal_thr=0.0):
    """Long-only vol-managed. Only rebalance if |Δw_target| > rebal_thr."""
    target_d = target_vol_ann / np.sqrt(365)
    target_w = (target_d / sigma_pred).clip(upper=MAX_W, lower=MIN_W)
    # Apply rebalance hysteresis
    held = []
    current = float(target_w.iloc[0])
    for tw in target_w.values:
        if abs(tw - current) > rebal_thr:
            current = float(tw)
        held.append(current)
    w = pd.Series(held, index=sigma_pred.index)
    w_lag = w.shift(1).fillna(0.0)
    turn = (w - w_lag).abs()
    r_gross = w * returns
    r_net = r_gross - fee * turn
    return w, turn, r_gross, r_net


def metrics(returns):
    r = returns.dropna()
    n = len(r)
    if n < 10: return None
    m, s = float(r.mean()), float(r.std(ddof=1))
    sh = np.sqrt(365) * m / s if s > 0 else 0.0
    dn = r[r < 0]
    so = np.sqrt(365) * m / dn.std(ddof=1) if len(dn) > 1 else 0.0
    eq = (1 + r).cumprod()
    tr = float(eq.iloc[-1] - 1)
    ar = float((1 + tr) ** (365 / n) - 1)
    av = float(s * np.sqrt(365))
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    cal = ar / abs(mdd) if mdd < 0 else 0
    return dict(n=n, tot_ret=tr, ann_ret=ar, ann_vol=av,
                sharpe=sh, sortino=so, mdd=mdd, calmar=cal)


def main():
    # Load all model forecasts (E1 walk-forward)
    forecasts = {}
    for f in ['HAR-RS-DOW', 'HAR-WE', 'HAR-RS-Q-WE-X']:
        df = pd.read_parquet(f'{PROJECT}/outputs/tables/E1_walkforward_per_day_{f}.parquet')
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        forecasts[f] = df

    MODELS = list(forecasts.keys())
    TARGETS = [0.25, 0.35, 0.45, 0.55]  # annualized vol target
    THRS = [0.0, 0.05, 0.10]             # min |Δw| to trigger rebalance
    n_dates = len(forecasts[MODELS[0]])

    results = []
    for mdl in MODELS:
        df = forecasts[mdl]
        sig = np.sqrt(np.exp(df['mu_pred']))
        ret = df['r_actual']
        for tv in TARGETS:
            for thr in THRS:
                for fee_name, fee_rate in FEES.items():
                    w, turn, rg, rn = vol_managed_lo(sig, ret, tv, fee_rate, thr)
                    m = metrics(rn)
                    if m:
                        m.update(strategy='vol-managed', model=mdl,
                                  target_vol=tv, threshold=thr,
                                  fee_label=fee_name, fee_rate=fee_rate,
                                  avg_turnover=float(turn.mean()),
                                  avg_weight=float(w.mean()),
                                  pct_invested=float((w > 0.1).mean()))
                        results.append(m)

    # Baseline 1: buy-and-hold (free, single trade)
    bh_ret = forecasts['HAR-RS-DOW']['r_actual']
    bh_m = metrics(bh_ret)
    bh_m.update(strategy='buy-and-hold', model='-', target_vol=np.nan,
                 threshold=np.nan, fee_label='none', fee_rate=0,
                 avg_turnover=0.0, avg_weight=1.0, pct_invested=1.0)
    results.append(bh_m)

    # Baseline 2: DCA weekly (€10/week, free SEPA deposit + 0.25% taker)
    # We measure time-weighted returns of the holdings
    n = len(bh_ret)
    weekly_idx = np.arange(0, n, 7)
    units_bought = np.zeros(n)
    # Each week buy €10 worth at that day's close. Track units, not euros.
    # Cumulative euro value of holdings each day.
    units = 0.0
    holdings_eur = []
    starting_btc_price = 100.0
    btc_price_series = starting_btc_price * (1 + bh_ret).cumprod()
    contributions = 0.0
    contribution_series = pd.Series(0.0, index=bh_ret.index)
    contribution_series.iloc[weekly_idx] = 10.0
    cash_invested = 0.0
    cost_basis = 0.0
    for i in range(n):
        # Buy at start of day (use yesterday's close = today's open approx)
        if i in weekly_idx:
            # taker fee on each €10 purchase
            buy_amount_after_fee = 10.0 * (1 - 0.0025)
            units_new = buy_amount_after_fee / btc_price_series.iloc[i]
            units += units_new
            cash_invested += 10.0
        # End-of-day value of holdings
        holdings_eur.append(units * btc_price_series.iloc[i])
    portfolio_value = pd.Series(holdings_eur, index=bh_ret.index)
    # Money-weighted return (rough)
    tot_invested = cash_invested
    final_value = portfolio_value.iloc[-1]
    # Daily returns of portfolio (for Sharpe; assuming the cash sit-out doesn't count)
    # Use buy-and-hold's vol/returns as approximation since position is in BTC
    dca_m = metrics(bh_ret)  # similar daily return profile, just smaller absolute
    if dca_m:
        # Override total return with actual money-weighted
        dca_m['tot_ret'] = (final_value / tot_invested) - 1
        years = n / 365
        dca_m['ann_ret'] = (final_value / tot_invested) ** (1 / years) - 1
        dca_m.update(strategy='DCA-weekly-€10', model='-',
                      target_vol=np.nan, threshold=np.nan,
                      fee_label='taker_25bps', fee_rate=0.0025,
                      avg_turnover=np.nan, avg_weight=np.nan, pct_invested=1.0)
        results.append(dca_m)

    # Save grid
    dfr = pd.DataFrame(results)
    dfr.to_csv(f'{PROJECT}/outputs/tables/G2_optimization_grid.csv', index=False)

    # Print ranked output
    print("=" * 100)
    print("G2 — Long-only vol-managed grid (Bitvavo realistic fees, OOS 2022-10 t/m 2026-05)")
    print("=" * 100)

    # Baseline summary first
    print("\nBASELINES:")
    print(f"  Buy-and-hold       : "
          f"TotRet {bh_m['tot_ret']*100:+.1f}%, AnnRet {bh_m['ann_ret']*100:+.1f}%, "
          f"Sharpe {bh_m['sharpe']:.2f}, MDD {bh_m['mdd']*100:.1f}%")
    print(f"  DCA-weekly-€10     : "
          f"TotRet {dca_m['tot_ret']*100:+.1f}%, AnnRet {dca_m['ann_ret']*100:+.1f}%, "
          f"Sharpe {dca_m['sharpe']:.2f}, MDD {dca_m['mdd']*100:.1f}%")

    # Top 10 by Sharpe
    print("\n" + "-" * 100)
    print("TOP-10 vol-managed by net Sharpe:")
    print("-" * 100)
    cols = ['model', 'target_vol', 'threshold', 'fee_label', 'tot_ret',
             'ann_ret', 'ann_vol', 'sharpe', 'mdd', 'avg_weight', 'avg_turnover']
    top_sharpe = dfr[dfr['strategy'] == 'vol-managed'].sort_values('sharpe', ascending=False).head(10)
    print(f"{'Model':<14}{'TgtVol':>7}{'Thr':>6}{'Fee':>14}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>8}{'MDD':>9}{'AvgW':>7}{'Turn':>7}")
    for _, r in top_sharpe.iterrows():
        print(f"{r['model']:<14}{r['target_vol']:>6.0%}{r['threshold']:>5.0%}"
              f"{r['fee_label']:>14}{r['tot_ret']*100:>+8.1f}%{r['ann_ret']*100:>+8.1f}%"
              f"{r['ann_vol']*100:>8.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>+8.1f}%"
              f"{r['avg_weight']:>7.2f}{r['avg_turnover']:>7.3f}")

    print("\n" + "-" * 100)
    print("TOP-5 vol-managed by total return:")
    print("-" * 100)
    top_ret = dfr[dfr['strategy'] == 'vol-managed'].sort_values('tot_ret', ascending=False).head(5)
    print(f"{'Model':<14}{'TgtVol':>7}{'Thr':>6}{'Fee':>14}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>8}{'MDD':>9}{'AvgW':>7}{'Turn':>7}")
    for _, r in top_ret.iterrows():
        print(f"{r['model']:<14}{r['target_vol']:>6.0%}{r['threshold']:>5.0%}"
              f"{r['fee_label']:>14}{r['tot_ret']*100:>+8.1f}%{r['ann_ret']*100:>+8.1f}%"
              f"{r['ann_vol']*100:>8.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>+8.1f}%"
              f"{r['avg_weight']:>7.2f}{r['avg_turnover']:>7.3f}")

    print("\n" + "-" * 100)
    print("BEST under risk constraints (MDD ≥ -50%):")
    print("-" * 100)
    constrained = dfr[(dfr['strategy'] == 'vol-managed') & (dfr['mdd'] >= -0.50)].sort_values('sharpe', ascending=False).head(5)
    for _, r in constrained.iterrows():
        print(f"{r['model']:<14}{r['target_vol']:>6.0%}{r['threshold']:>5.0%}"
              f"{r['fee_label']:>14}{r['tot_ret']*100:>+8.1f}%{r['ann_ret']*100:>+8.1f}%"
              f"{r['ann_vol']*100:>8.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>+8.1f}%"
              f"{r['avg_weight']:>7.2f}{r['avg_turnover']:>7.3f}")

    print("\n" + "-" * 100)
    print("BEST under risk constraints (MDD ≥ -35% — conservatief):")
    print("-" * 100)
    constrained = dfr[(dfr['strategy'] == 'vol-managed') & (dfr['mdd'] >= -0.35)].sort_values('sharpe', ascending=False).head(5)
    for _, r in constrained.iterrows():
        print(f"{r['model']:<14}{r['target_vol']:>6.0%}{r['threshold']:>5.0%}"
              f"{r['fee_label']:>14}{r['tot_ret']*100:>+8.1f}%{r['ann_ret']*100:>+8.1f}%"
              f"{r['ann_vol']*100:>8.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>+8.1f}%"
              f"{r['avg_weight']:>7.2f}{r['avg_turnover']:>7.3f}")

    print()
    print(f"Total configs evaluated: {len(dfr)}")
    print(f"Saved: outputs/tables/G2_optimization_grid.csv")


if __name__ == '__main__': main()
