"""H1 — Multi-frequency × multi-strategy comparison op BTC-EUR.

Strategieën (allemaal long-only, scale-invariant op max_alloc=0.90):
  1. Buy-and-hold
  2. Trend MA50  (50-bar moving average crossover)
  3. Trend MA200 (200-bar moving average crossover)
  4. Momentum-20 (positief rolling 20-bar return → in BTC)
  5. RSI-14      (RSI < 30 → in BTC, anders cash) — mean-reversion proxy

Frequenties: 1m, 5m, 15m, 1h, 1d. OOS-periode 2022-10 → 2026-05 voor fair comparison.
Fees: 15 bps maker (optimistisch — bij 1m/5m kan dit slechter zijn door spread).
Account: €110, 5% rebalance deadband, €5 min trade.
"""
from __future__ import annotations
import sys; sys.path.insert(0, 'src')
import time
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

FEE = 0.0015
MIN_TRADE = 5.0
MAX_W = 0.90
DEADBAND = 0.05
START = '2022-10-01'
END = '2026-05-15'
INIT = 110.0


def load_freq(freq: str) -> pd.DataFrame:
    df = pd.read_parquet(P / f'data/raw/candles_BTC-EUR_{freq}.parquet')
    df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('ts_dt').sort_index()
    df = df.loc[START:END]
    return df


# -------- Strategies (return target_weight series) --------
def s_buyhold(returns):
    return pd.Series(MAX_W, index=returns.index, dtype=float)

def s_trend(returns, ma_bars):
    price = (1 + returns).cumprod()
    ma = price.rolling(ma_bars, min_periods=max(10, ma_bars // 5)).mean()
    w = pd.Series(np.where(price > ma, MAX_W, 0.0), index=returns.index)
    return w.shift(1).fillna(0.0)  # signal from previous bar, no lookahead

def s_momentum(returns, n_bars):
    mom = returns.rolling(n_bars).sum()
    w = pd.Series(np.where(mom > 0, MAX_W, 0.0), index=returns.index)
    return w.shift(1).fillna(0.0)

def s_rsi(returns, n_bars=14, oversold=30):
    gain = returns.where(returns > 0, 0).rolling(n_bars).sum()
    loss = (-returns.where(returns < 0, 0)).rolling(n_bars).sum()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    w = pd.Series(np.where(rsi < oversold, MAX_W, 0.0), index=returns.index)
    return w.shift(1).fillna(0.0)


# -------- Portfolio simulator (long-only, deadband+min-trade) --------
def simulate(weights: pd.Series, returns: pd.Series, init=INIT,
              fee=FEE, min_trade=MIN_TRADE, deadband=DEADBAND):
    """Vectorized-ish portfolio sim. Returns (equity_series, total_fees, n_rebal)."""
    n = len(returns)
    pv, btc, eur = init, 0.0, init
    fees = 0.0; n_rebal = 0; n_skip = 0
    weights_arr = weights.values
    rets_arr = returns.values
    pvs = np.empty(n)
    for i in range(n):
        btc *= (1 + rets_arr[i])
        pv = btc + eur
        current_w = btc / pv if pv > 0 else 0
        target_w = float(weights_arr[i])
        if abs(target_w - current_w) > deadband:
            target_eur = target_w * pv
            d = target_eur - btc
            if abs(d) >= min_trade:
                fee_p = abs(d) * fee
                btc = target_eur; eur = pv - target_eur - fee_p
                pv = btc + eur
                fees += fee_p; n_rebal += 1
            else:
                n_skip += 1
        pvs[i] = pv
    return pd.Series(pvs, index=returns.index), fees, n_rebal, n_skip


def metrics_freq(pv: pd.Series, init: float, freq: str):
    r = pv.pct_change().dropna()
    n = len(r)
    if n < 10 or pv.iloc[-1] <= 0: return None
    final = float(pv.iloc[-1])
    
    # Annualization factor depends on frequency
    bars_per_year = {'1m': 365*24*60, '5m': 365*24*12, '15m': 365*24*4,
                     '1h': 365*24, '1d': 365}[freq]
    actual_years = n / bars_per_year
    
    tot_ret = (final/init) - 1
    ann_ret = (final/init) ** (1/actual_years) - 1 if actual_years > 0 else 0
    ann_vol = float(r.std(ddof=1)) * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    mdd = float(((pv-pv.cummax())/pv.cummax()).min())
    return dict(final=final, tot_ret=tot_ret, ann_ret=ann_ret,
                ann_vol=ann_vol, sharpe=sharpe, mdd=mdd,
                calmar=ann_ret/abs(mdd) if mdd<0 else 0)


STRATEGIES = [
    ('B&H', s_buyhold, {}),
    ('Trend MA50', s_trend, {'ma_bars': 50}),
    ('Trend MA200', s_trend, {'ma_bars': 200}),
    ('Momentum-20', s_momentum, {'n_bars': 20}),
    ('RSI-14<30', s_rsi, {'n_bars': 14}),
]

FREQS = ['1m', '5m', '15m', '1h', '1d']


def main():
    print("=" * 120)
    print("H1 — Multi-frequency × Multi-strategy op BTC-EUR (€110, 15 bps fees)")
    print("=" * 120)
    
    all_results = []
    for freq in FREQS:
        t0 = time.time()
        candles = load_freq(freq)
        returns = candles['close'].pct_change().dropna()
        print(f"\n{'-'*120}")
        print(f"  Frequentie: {freq:<5} | n_bars={len(returns):,} | "
              f"periode={returns.index[0]} → {returns.index[-1]} | load_time={time.time()-t0:.1f}s")
        print(f"{'-'*120}")
        print(f"  {'Strategy':<15}{'Final':>9}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>8}"
              f"{'MDD':>9}{'#Reb':>8}{'#Skp':>8}{'Fees':>8}{'Fee%':>7}")
        
        for s_name, s_fn, s_kw in STRATEGIES:
            t1 = time.time()
            weights = s_fn(returns, **s_kw)
            pv, fees, nr, ns = simulate(weights, returns, init=INIT)
            m = metrics_freq(pv, INIT, freq)
            if m is None: continue
            sim_time = time.time() - t1
            fee_pct = fees / INIT * 100
            all_results.append({'freq': freq, 'strategy': s_name,
                                 'n_bars': len(returns), 'sim_time': sim_time,
                                 'fees': fees, 'n_rebal': nr, 'n_skip': ns, **m})
            print(f"  {s_name:<15}{m['final']:>9.2f}{m['tot_ret']*100:>+8.1f}%"
                  f"{m['ann_ret']*100:>+8.1f}%{m['ann_vol']*100:>8.1f}%"
                  f"{m['sharpe']:>+8.2f}{m['mdd']*100:>+8.1f}%{nr:>8d}{ns:>8d}"
                  f"{fees:>8.2f}{fee_pct:>6.1f}%")
    
    # Save full grid
    df = pd.DataFrame(all_results)
    df.to_csv(P / 'outputs/tables/H1_multifreq_results.csv', index=False)
    print(f"\nGesaved: outputs/tables/H1_multifreq_results.csv ({len(df)} configs)")
    
    # Top-5 ranking
    print("\n" + "=" * 80)
    print("TOP-5 by Sharpe (over alle freq × strategy combinaties)")
    print("=" * 80)
    top = df.sort_values('sharpe', ascending=False).head(10)
    print(top[['freq', 'strategy', 'final', 'ann_ret', 'sharpe', 'mdd', 'n_rebal', 'fees']]
            .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Heatmap: Sharpe per (freq, strategy)
    pivot_sharpe = df.pivot(index='strategy', columns='freq', values='sharpe')
    pivot_sharpe = pivot_sharpe[FREQS]  # ordered
    pivot_ret = df.pivot(index='strategy', columns='freq', values='ann_ret') * 100
    pivot_ret = pivot_ret[FREQS]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, cmap in [
        (axes[0], pivot_sharpe, 'Net Sharpe per (freq, strategy)', 'RdYlGn'),
        (axes[1], pivot_ret, 'Annualized Return (%) per (freq, strategy)', 'RdYlGn'),
    ]:
        im = ax.imshow(data.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(FREQS)))
        ax.set_xticklabels(FREQS)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        ax.set_title(title, fontweight='bold')
        for i, row in enumerate(data.values):
            for j, val in enumerate(row):
                if pd.notna(val):
                    color = 'white' if abs(val) > data.values.max()*0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=10)
        fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    fig.savefig(P / 'outputs/figures/H1_multifreq_heatmap.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFiguur: outputs/figures/H1_multifreq_heatmap.png")


if __name__ == '__main__':
    main()
