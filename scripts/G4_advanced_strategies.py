"""G4 — Geavanceerde strategie-design-space voor BTC-EUR vol-managed.

Vier strategie-families bovenop pure vol-managed:
  - Threshold-trigger (τ ∈ {5%, 10%, 15%, 20%})
  - Pure trend-filter (MA50, MA200 alleen)
  - Vol-managed + trend filter (combinatie)
  - Threshold-trigger + trend filter

Doel: vinden of er iets bestaat dat statistisch én economisch
buy-and-hold verslaat (Sharpe én total return).
"""
from __future__ import annotations
import sys; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')
FEE = 0.0015
MIN_TRADE = 5.0
MAX_ALLOC = 0.90
TARGET_VOL = 0.35
MODEL = 'HAR-RS-DOW'  # G3-winnaar bij biweekly


def stats(pv, init):
    r = pv.pct_change().dropna(); n = len(r)
    final = float(pv.iloc[-1])
    if final <= 0: return None
    ar = (final/init)**(365/n) - 1
    av = float(r.std(ddof=1)) * np.sqrt(365)
    sh = ar/av if av > 0 else 0
    dn = r[r<0]
    so = ar/(dn.std(ddof=1)*np.sqrt(365)) if len(dn)>1 else 0
    mdd = float(((pv-pv.cummax())/pv.cummax()).min())
    pct_inv = float((pv.pct_change().abs() > 1e-9).mean())  # rough "active" pct
    return dict(final=final, ann_ret=ar, ann_vol=av, sharpe=sh, sortino=so,
                 mdd=mdd, calmar=ar/abs(mdd) if mdd<0 else 0)


def base_simulator(target_w_series, ret, init, rebal_func, **kwargs):
    """Generieke simulator. rebal_func(i, target_w_today, current_w) → bool of rebalance."""
    n = len(target_w_series)
    pv, btc, eur = init, 0.0, init
    pvs, fees, n_rebal, n_skip = [], 0.0, 0, 0
    for i in range(n):
        btc = btc * (1 + ret.iloc[i]); pv = btc + eur
        current_w = btc / pv if pv > 0 else 0
        target_w = float(target_w_series.iloc[i])
        do_rebal, eff_target_w = rebal_func(i, target_w, current_w, **kwargs)
        if do_rebal:
            target_eur = eff_target_w * pv
            d = target_eur - btc
            if abs(d) >= MIN_TRADE:
                fee_p = abs(d) * FEE
                btc, eur = target_eur, pv - target_eur - fee_p
                pv = btc + eur
                fees += fee_p; n_rebal += 1
            else:
                n_skip += 1
        pvs.append(pv)
    return pd.Series(pvs, index=target_w_series.index), fees, n_rebal, n_skip


# Strategy rebal rules
def r_daily(i, tw, cw): return True, tw
def r_every_n(i, tw, cw, N=14): return (i % N == 0), tw
def r_threshold(i, tw, cw, tau=0.10): return abs(tw - cw) > tau, tw
def r_trend_only(i, tw, cw, trend=None, max_w=0.90):
    """Pure trend: w=max_w if in trend, 0 otherwise. Rebalance always."""
    eff = max_w if trend[i] else 0.0
    return abs(eff - cw) > 0.05, eff  # 5% deadband to reduce churn
def r_voltrend_n(i, tw, cw, N=14, trend=None):
    eff = tw if trend[i] else 0.0
    return (i % N == 0), eff
def r_threshold_trend(i, tw, cw, tau=0.10, trend=None):
    eff = tw if trend[i] else 0.0
    return abs(eff - cw) > tau, eff


def main():
    df = pd.read_parquet(f'{P}/outputs/tables/E1_walkforward_per_day_{MODEL}.parquet')
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    sig = np.sqrt(np.exp(df['mu_pred']))
    ret = df['r_actual']
    
    # Construct target weight series (vol-managed)
    target_d = TARGET_VOL / np.sqrt(365)
    tw_series = (target_d / sig).clip(upper=MAX_ALLOC, lower=0.0)
    
    # Construct trend signals (BTC price reconstructed from returns)
    price = (1 + ret).cumprod()
    ma50 = price.rolling(50, min_periods=10).mean()
    ma200 = price.rolling(200, min_periods=50).mean()
    in_ma50 = (price > ma50).fillna(True).values
    in_ma200 = (price > ma200).fillna(True).values
    
    print(f"\nModel = {MODEL} (G3-winnaar bij biweekly)")
    print(f"Price trends: {in_ma50.mean()*100:.1f}% van dagen boven MA50, "
          f"{in_ma200.mean()*100:.1f}% boven MA200")
    
    strategies = [
        # (label, target_w_series, rebal_func, kwargs)
        ('Buy-and-hold',          pd.Series(MAX_ALLOC, index=sig.index), r_daily, {}),
        ('VolM daily',            tw_series, r_daily, {}),
        ('VolM 14d (G3 winner)',  tw_series, r_every_n, {'N': 14}),
        ('VolM 22d',              tw_series, r_every_n, {'N': 22}),
        ('VolM 30d',              tw_series, r_every_n, {'N': 30}),
        ('Threshold τ=5%',        tw_series, r_threshold, {'tau': 0.05}),
        ('Threshold τ=10%',       tw_series, r_threshold, {'tau': 0.10}),
        ('Threshold τ=15%',       tw_series, r_threshold, {'tau': 0.15}),
        ('Threshold τ=20%',       tw_series, r_threshold, {'tau': 0.20}),
        ('Trend MA50 only',       tw_series, r_trend_only, {'trend': in_ma50}),
        ('Trend MA200 only',      tw_series, r_trend_only, {'trend': in_ma200}),
        ('VolM 14d + MA50',       tw_series, r_voltrend_n, {'N': 14, 'trend': in_ma50}),
        ('VolM 14d + MA200',      tw_series, r_voltrend_n, {'N': 14, 'trend': in_ma200}),
        ('VolM 22d + MA50',       tw_series, r_voltrend_n, {'N': 22, 'trend': in_ma50}),
        ('VolM 22d + MA200',      tw_series, r_voltrend_n, {'N': 22, 'trend': in_ma200}),
        ('Thr 10% + MA50',        tw_series, r_threshold_trend, {'tau': 0.10, 'trend': in_ma50}),
        ('Thr 10% + MA200',       tw_series, r_threshold_trend, {'tau': 0.10, 'trend': in_ma200}),
        ('Thr 15% + MA50',        tw_series, r_threshold_trend, {'tau': 0.15, 'trend': in_ma50}),
        ('Thr 15% + MA200',       tw_series, r_threshold_trend, {'tau': 0.15, 'trend': in_ma200}),
    ]
    
    print("\n" + "="*108)
    print(f"G4 — Strategy comparison op €110 (BTC-EUR, {MODEL}, "
           f"target_vol={TARGET_VOL:.0%}, max_alloc={MAX_ALLOC:.0%}, fee=15bps)")
    print("="*108)
    print(f"{'Strategy':<25}{'Final':>9}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}"
           f"{'Sharpe':>8}{'Sort':>7}{'MDD':>9}{'Cal':>6}{'#Reb':>7}{'#Skp':>7}{'Fees':>7}")
    print("-"*108)
    
    all_results = []
    for label, twin, fn, kw in strategies:
        # Buy-and-hold is special: always at MAX_ALLOC, no rebalancing
        if label == 'Buy-and-hold':
            pv = 110 * MAX_ALLOC * (1 + ret).cumprod() + 110 * (1 - MAX_ALLOC)
            fees, nr, ns = 0.0, 0, 0
        else:
            pv, fees, nr, ns = base_simulator(twin, ret, 110, fn, **kw)
        s = stats(pv, 110)
        if s is None: continue
        all_results.append({'label': label, 'pv': pv, 'fees': fees,
                             'n_rebal': nr, 'n_skip': ns, **s})
        print(f"{label:<25}{s['final']:>9.2f}{(s['final']/110-1)*100:>+8.1f}%"
              f"{s['ann_ret']*100:>+8.1f}%{s['ann_vol']*100:>8.1f}%"
              f"{s['sharpe']:>8.2f}{s['sortino']:>7.2f}{s['mdd']*100:>+8.1f}%"
              f"{s['calmar']:>6.2f}{nr:>7d}{ns:>7d}{fees:>7.2f}")
    
    print("-"*108)
    
    # Top 5 by Sharpe and by total return
    by_sh = sorted(all_results, key=lambda r: r['sharpe'], reverse=True)[:5]
    by_ret = sorted(all_results, key=lambda r: r['final'], reverse=True)[:5]
    
    print(f"\nTop-5 by Sharpe:")
    for r in by_sh:
        print(f"  {r['label']:<25} Sharpe={r['sharpe']:.3f}, AnnRet={r['ann_ret']*100:+.1f}%, "
              f"MDD={r['mdd']*100:+.1f}%")
    
    print(f"\nTop-5 by Total Return:")
    for r in by_ret:
        print(f"  {r['label']:<25} Final=€{r['final']:.2f}, AnnRet={r['ann_ret']*100:+.1f}%, "
              f"Sharpe={r['sharpe']:.2f}")
    
    # Save CSV
    out = []
    for r in all_results:
        out.append({k: v for k, v in r.items() if k not in ('pv',)})
    pd.DataFrame(out).to_csv(f'{P}/outputs/tables/G4_advanced_strategies.csv', index=False)
    
    # Plot top-5 + buy-and-hold
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    
    plot_strategies = ['Buy-and-hold', 'VolM daily']
    # Add unique top performers
    seen = set(plot_strategies)
    for r in by_sh:
        if r['label'] not in seen:
            plot_strategies.append(r['label'])
            seen.add(r['label'])
        if len(plot_strategies) >= 6: break
    
    colors = ['#6b7280', '#dc2626', '#059669', '#7c3aed', '#f59e0b', '#0891b2', '#be185d']
    label_to_pv = {r['label']: r['pv'] for r in all_results}
    
    for label, color in zip(plot_strategies, colors):
        if label in label_to_pv:
            pv = label_to_pv[label]
            axes[0].plot(pv.index, pv.values, lw=1.5, color=color, label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Portfolio €')
    axes[0].set_title(f'Top strategies vs B&H on €110 ({MODEL})', fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Drawdown panel
    for label, color in zip(plot_strategies, colors):
        if label in label_to_pv:
            pv = label_to_pv[label]
            dd = (pv - pv.cummax()) / pv.cummax()
            axes[1].fill_between(dd.index, dd.values * 100, 0, alpha=0.3, color=color, label=label)
    axes[1].set_ylabel('Drawdown %')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f'{P}/outputs/figures/G4_advanced_strategies.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: outputs/figures/G4_advanced_strategies.png")
    print(f"       outputs/tables/G4_advanced_strategies.csv")


if __name__ == '__main__':
    main()
