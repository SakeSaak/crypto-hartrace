"""H2 — Klassieke TA-strategieën + HAR-leveraged varianten op daily BTC-EUR.

Doel: zoekt het echte rendementsplafond op €110 met G2-realistische Bitvavo-fees.
Testen op gelijke OOS-periode (2022-10-01 → 2026-05-14) zodat alles vergelijkbaar.

Strategieën:
  Benchmarks:
    - B&H (passive baseline)
  Klassieke TA:
    - Trend MA50 (current bot strategy)
    - Bollinger Bands mean-reversion (BB-MR)
    - Bollinger Bands breakout (BB-BO)
    - MACD crossover (12/26/9)
    - ATR-trailing stop op trend MA50
    - Donchian channels (entry 20d, exit 10d)
    - RSI-bounce (cross up uit oversold)
  HAR-leveraged (nieuwe):
    - HAR vol-regime entry filter (trend + only-if-low-vol)
    - HAR vol-targeted sizing binnen trend
  Hybriden:
    - Trend MA50 + Bollinger pullback re-entry
"""
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

FEE = 0.0015          # 15 bps maker (Bitvavo retail)
MIN_TRADE = 5.0       # €5 minimum trade size (Bitvavo)
MAX_W = 0.90          # max 90% allocation (10% EUR buffer)
DEADBAND = 0.05       # rebalance bij |Δw| > 5%
INIT = 110.0          # account size
TARGET_VOL_ANN = 0.35 # voor HAR vol-targeting


# === Strategy implementations (all return target_weight series, lag-1 to avoid lookahead) ===

def lag1(w_series):
    """Apply day-t signal to day-t+1 position to avoid look-ahead bias."""
    return w_series.shift(1).fillna(0.0)


def s_buyhold(prices, returns, sigma_pred=None):
    return pd.Series(MAX_W, index=returns.index, dtype=float)


def s_trend_ma50(prices, returns, sigma_pred=None, ma_window=50):
    ma = prices.rolling(ma_window, min_periods=10).mean()
    w = pd.Series(np.where(prices > ma, MAX_W, 0.0), index=returns.index)
    return lag1(w)


def s_bollinger_mr(prices, returns, sigma_pred=None, n=20, k=2.0):
    """Mean-reversion: long when price < lower band, exit when crosses middle."""
    ma = prices.rolling(n).mean()
    std = prices.rolling(n).std()
    lower = ma - k * std
    w = np.zeros(len(prices))
    state = False
    for i in range(len(prices)):
        if pd.isna(ma.iloc[i]):
            continue
        if not state and prices.iloc[i] < lower.iloc[i]:
            state = True
        elif state and prices.iloc[i] > ma.iloc[i]:
            state = False
        w[i] = MAX_W if state else 0.0
    return lag1(pd.Series(w, index=returns.index))


def s_bollinger_bo(prices, returns, sigma_pred=None, n=20, k=2.0):
    """Breakout: long when price > upper band, exit when crosses below middle."""
    ma = prices.rolling(n).mean()
    std = prices.rolling(n).std()
    upper = ma + k * std
    w = np.zeros(len(prices))
    state = False
    for i in range(len(prices)):
        if pd.isna(ma.iloc[i]):
            continue
        if not state and prices.iloc[i] > upper.iloc[i]:
            state = True
        elif state and prices.iloc[i] < ma.iloc[i]:
            state = False
        w[i] = MAX_W if state else 0.0
    return lag1(pd.Series(w, index=returns.index))


def s_macd(prices, returns, sigma_pred=None, fast=12, slow=26, sig_n=9):
    """MACD crossover: long when MACD > signal line."""
    ema_f = prices.ewm(span=fast, adjust=False).mean()
    ema_s = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=sig_n, adjust=False).mean()
    w = pd.Series(np.where(macd > sig, MAX_W, 0.0), index=returns.index)
    return lag1(w)


def s_atr_stop(prices, returns, sigma_pred=None, ma_window=50, atr_window=14, atr_mult=3.0):
    """Trend MA50 entry, ATR-trailing stop exit."""
    ma = prices.rolling(ma_window, min_periods=10).mean()
    tr = prices.diff().abs()
    atr = tr.rolling(atr_window).mean()
    w = np.zeros(len(prices))
    state = False
    stop = -np.inf
    for i in range(len(prices)):
        if pd.isna(ma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if not state and prices.iloc[i] > ma.iloc[i]:
            state = True
            stop = prices.iloc[i] - atr_mult * atr.iloc[i]
        elif state:
            new_stop = prices.iloc[i] - atr_mult * atr.iloc[i]
            stop = max(stop, new_stop)
            if prices.iloc[i] < stop:
                state = False
                stop = -np.inf
        w[i] = MAX_W if state else 0.0
    return lag1(pd.Series(w, index=returns.index))


def s_donchian(prices, returns, sigma_pred=None, entry_n=20, exit_n=10):
    """Donchian channels (Turtle): break entry-high enter, break exit-low exit."""
    entry_high = prices.rolling(entry_n).max()
    exit_low = prices.rolling(exit_n).min()
    w = np.zeros(len(prices))
    state = False
    for i in range(max(entry_n, exit_n), len(prices)):
        if not state and prices.iloc[i] > entry_high.iloc[i-1]:
            state = True
        elif state and prices.iloc[i] < exit_low.iloc[i-1]:
            state = False
        w[i] = MAX_W if state else 0.0
    return lag1(pd.Series(w, index=returns.index))


def s_rsi_bounce(prices, returns, sigma_pred=None, n=14, oversold=30, exit_level=50):
    """RSI bounce: enter when RSI crosses up out of oversold, exit when crosses 50."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    w = np.zeros(len(prices))
    state = False
    prev = None
    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue
        if not state and prev is not None and prev < oversold and r >= oversold:
            state = True
        elif state and r > exit_level:
            state = False
        prev = r
        w[i] = MAX_W if state else 0.0
    return lag1(pd.Series(w, index=returns.index))


def s_har_volregime_trend(prices, returns, sigma_pred, ma_window=50, vol_quantile=0.5):
    """HAR-leveraged: trend entry alleen als HAR-vol < expanding median (low-vol regime)."""
    ma = prices.rolling(ma_window, min_periods=10).mean()
    in_trend = (prices > ma).fillna(False)
    # Expanding quantile (real-time, no look-ahead)
    vol_thresh = sigma_pred.expanding(min_periods=50).quantile(vol_quantile)
    low_vol = (sigma_pred < vol_thresh).fillna(False)
    in_position = in_trend & low_vol
    w = pd.Series(np.where(in_position, MAX_W, 0.0), index=returns.index)
    return lag1(w)


def s_har_voltarget_trend(prices, returns, sigma_pred, ma_window=50):
    """HAR-leveraged: trend entry maar size = vol-targeted (within MAX_W cap)."""
    ma = prices.rolling(ma_window, min_periods=10).mean()
    in_trend = (prices > ma).fillna(False).values
    target_d = TARGET_VOL_ANN / np.sqrt(365)
    vol_w = (target_d / sigma_pred).clip(upper=MAX_W, lower=0.0).values
    w_arr = np.where(in_trend, vol_w, 0.0)
    return lag1(pd.Series(w_arr, index=returns.index))


def s_trend_bb_reentry(prices, returns, sigma_pred=None, ma_window=50, bb_n=20, bb_k=2.0):
    """Hybride: in trend AND (above MA50 OR recently bounced from BB lower)."""
    ma50 = prices.rolling(ma_window, min_periods=10).mean()
    bb_ma = prices.rolling(bb_n).mean()
    bb_std = prices.rolling(bb_n).std()
    bb_lower = bb_ma - bb_k * bb_std
    in_trend = (prices > ma50).fillna(False)
    # When in trend, accept any price. When out of trend, accept if pulled back to BB lower
    # (early re-entry signal during temporary dip in larger uptrend regime)
    # Simplified: just trend OR (within 5% above MA50 AND price < BB lower)
    pullback = (prices < bb_lower) & (prices > ma50 * 0.95)
    in_position = in_trend | pullback
    w = pd.Series(np.where(in_position, MAX_W, 0.0), index=returns.index)
    return lag1(w)


# === Portfolio simulator (zelfde als G3/G4/H1) ===

def simulate(weights, returns, init=INIT, fee=FEE, min_trade=MIN_TRADE, deadband=DEADBAND):
    n = len(returns)
    pv, btc, eur = init, 0.0, init
    fees = 0.0; n_rebal = 0; n_skip = 0
    w_arr = weights.values; r_arr = returns.values
    pvs = np.empty(n)
    for i in range(n):
        btc *= (1 + r_arr[i])
        pv = btc + eur
        cw = btc/pv if pv > 0 else 0
        tw = float(w_arr[i])
        if abs(tw - cw) > deadband:
            te = tw * pv; d = te - btc
            if abs(d) >= min_trade:
                fp = abs(d) * fee
                btc = te; eur = pv - te - fp; pv = btc + eur
                fees += fp; n_rebal += 1
            else:
                n_skip += 1
        pvs[i] = pv
    return pd.Series(pvs, index=returns.index), fees, n_rebal, n_skip


def metrics(pv, init):
    r = pv.pct_change().dropna()
    n = len(r)
    if n < 10 or pv.iloc[-1] <= 0: return None
    final = float(pv.iloc[-1])
    ar = (final/init) ** (365/n) - 1
    av = float(r.std(ddof=1)) * np.sqrt(365)
    sh = ar/av if av > 0 else 0
    dn = r[r<0]
    so = ar/(dn.std(ddof=1)*np.sqrt(365)) if len(dn)>1 else 0
    mdd = float(((pv-pv.cummax())/pv.cummax()).min())
    return dict(final=final, tot_ret=final/init-1, ann_ret=ar, ann_vol=av,
                sharpe=sh, sortino=so, mdd=mdd,
                calmar=ar/abs(mdd) if mdd<0 else 0)


# === Main ===

# Load data: HAR forecasts + actual returns (uit E1 walk-forward)
df = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)
returns = df['r_actual']
prices = (1 + returns).cumprod() * 100  # arbitrary base
sigma_pred = np.sqrt(np.exp(df['mu_pred']))  # σ from log-RK forecast

print(f"OOS-periode: {returns.index.min().date()} → {returns.index.max().date()} (n={len(returns)})")
print(f"Setup: €{INIT}, fee={FEE*1e4:.0f}bps, max_alloc={MAX_W:.0%}, deadband={DEADBAND:.0%}, min_trade=€{MIN_TRADE}")

STRATS = [
    ('B&H',                      s_buyhold,             {}),
    ('Trend MA50',               s_trend_ma50,          {}),
    ('Bollinger MR',             s_bollinger_mr,        {}),
    ('Bollinger Breakout',       s_bollinger_bo,        {}),
    ('MACD',                     s_macd,                {}),
    ('ATR-stop on Trend MA50',   s_atr_stop,            {}),
    ('Donchian',                 s_donchian,            {}),
    ('RSI-Bounce',               s_rsi_bounce,          {}),
    ('HAR vol-regime + Trend',   s_har_volregime_trend, {}),
    ('HAR vol-target + Trend',   s_har_voltarget_trend, {}),
    ('Trend MA50 + BB pullback', s_trend_bb_reentry,    {}),
]

print("\n" + "="*125)
print(f"  {'Strategy':<27}{'Final':>9}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}"
      f"{'Sharpe':>8}{'Sortino':>9}{'MDD':>9}{'Calmar':>8}{'#Reb':>7}{'Fees':>8}")
print("="*125)

results = []
equity_curves = {}
for name, fn, kw in STRATS:
    weights = fn(prices, returns, sigma_pred, **kw)
    pv, fees, nr, ns = simulate(weights, returns)
    m = metrics(pv, INIT)
    if m is None: continue
    results.append({'strategy': name, 'fees': fees, 'n_rebal': nr, 'n_skip': ns, **m})
    equity_curves[name] = pv
    print(f"  {name:<27}{m['final']:>9.2f}{m['tot_ret']*100:>+8.1f}%"
          f"{m['ann_ret']*100:>+8.1f}%{m['ann_vol']*100:>8.1f}%"
          f"{m['sharpe']:>+8.2f}{m['sortino']:>+9.2f}{m['mdd']*100:>+8.1f}%"
          f"{m['calmar']:>8.2f}{nr:>7d}{fees:>8.2f}")

# Save full results
df_out = pd.DataFrame(results)
df_out.to_csv(P/'outputs/tables/H2_TA_strategies.csv', index=False)

# Ranking
print("\n" + "="*60)
print("Top-5 by Sharpe:")
print("="*60)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]:
    print(f"  {r['strategy']:<27} Sharpe={r['sharpe']:+.3f}  "
          f"AnnRet={r['ann_ret']*100:+5.1f}%  MDD={r['mdd']*100:+5.1f}%")

print("\nTop-5 by Total Return:")
print("="*60)
for r in sorted(results, key=lambda x: x['final'], reverse=True)[:5]:
    print(f"  {r['strategy']:<27} Final=€{r['final']:6.2f}  "
          f"Sharpe={r['sharpe']:+.2f}  MDD={r['mdd']*100:+5.1f}%")

# Plot top performers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
plot_set = ['B&H', 'Trend MA50']
# add unique top-5 sharpe performers
seen = set(plot_set)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    if r['strategy'] not in seen:
        plot_set.append(r['strategy'])
        seen.add(r['strategy'])
    if len(plot_set) >= 6: break

colors = ['#6b7280', '#dc2626', '#059669', '#7c3aed', '#f59e0b', '#0891b2', '#be185d']
for name, color in zip(plot_set, colors):
    if name in equity_curves:
        pv = equity_curves[name]
        sh = next(r['sharpe'] for r in results if r['strategy']==name)
        ax1.plot(pv.index, pv.values, lw=1.5, color=color, label=f'{name} (Sh={sh:.2f})')
        dd = (pv - pv.cummax()) / pv.cummax()
        ax2.fill_between(dd.index, dd.values*100, 0, alpha=0.3, color=color)

ax1.set_yscale('log'); ax1.set_ylabel('Portfolio €')
ax1.set_title('H2 — TA Strategies + HAR-leveraged variants on daily BTC-EUR (€110, 15bps)',
                fontweight='bold')
ax1.legend(loc='upper left', fontsize=9); ax1.grid(True, alpha=0.3, which='both')
ax2.set_ylabel('Drawdown %'); ax2.set_xlabel('Date'); ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H2_TA_strategies.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Resultaten: outputs/tables/H2_TA_strategies.csv")
print(f"✓ Figuur:     outputs/figures/H2_TA_strategies.png")
