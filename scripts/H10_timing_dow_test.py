"""H10 — Timing-test: optimaal cron-tijdstip + day-of-week filtering.

Twee vragen:
  Q1: Maakt het cron-tijdstip uit? Test 0/4/8/12/16/20 UTC.
  Q2: Helpen DOW-inzichten? Test:
      - Skip Saturday (laagste vol — wijde spreads in werkelijkheid)
      - Skip weekend (Sat + Sun)
      - DOW-aware sizing (kleinere positie op hoge-vol dagen)

Strategy: Pure Trend MA50 op daily close (zoals huidige bot).
OOS: 2022-10-01 → 2026-05-14
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')
OOS_START = pd.Timestamp('2022-10-01')
OOS_END   = pd.Timestamp('2026-05-14')

def load_1h():
    df = pd.read_parquet(P/'data/raw/candles_BTC-EUR_1h.parquet')
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('ts')[['close']].sort_index()
    return df.loc[OOS_START:OOS_END]

def backtest_trend(prices: pd.Series, day_filter=None, position_scaler=None,
                   ma_window: int = 50, fee: float = 0.0015, deadband: float = 0.05):
    """Pure trend backtest met optionele day-filter en position-scaler.
    
    day_filter:        callable(timestamp) → True als we mogen handelen die dag
    position_scaler:   callable(timestamp) → multiplier 0-1 op target weight
    """
    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    cash, btc = 10_000.0, 0.0
    equity = pd.Series(index=prices.index, dtype=float)
    n_trades, fees_total = 0, 0.0
    
    for i, t in enumerate(prices.index):
        p = prices.iloc[i]
        wealth = cash + btc * p
        equity.iloc[i] = wealth
        
        if pd.isna(ma.iloc[i]):
            continue
        
        in_trend = 1 if p > ma.iloc[i] else 0
        target_w = 0.90 * in_trend
        if position_scaler:
            target_w *= position_scaler(t)
        
        current_w = (btc * p) / wealth if wealth > 0 else 0
        can_trade = (day_filter(t) if day_filter else True) and i > 0
        
        if can_trade and abs(target_w - current_w) > deadband:
            delta = wealth * target_w - btc * p
            fee_eur = abs(delta) * fee
            if delta > 0:
                btc += (delta - fee_eur) / p
                cash -= delta
            else:
                btc += delta / p
                cash -= delta - fee_eur
            cash -= fee_eur
            fees_total += fee_eur
            n_trades += 1
    
    equity = equity.dropna()
    log_ret = np.log(equity/equity.shift(1)).dropna()
    n_yrs = (equity.index[-1]-equity.index[0]).days / 365.25
    
    return {
        'final': equity.iloc[-1],
        'ann_return_pct': ((equity.iloc[-1]/10_000)**(1/n_yrs)-1)*100,
        'sharpe': (log_ret.mean()*365)/(log_ret.std()*np.sqrt(365)) if log_ret.std() > 0 else 0,
        'mdd_pct': ((equity/equity.cummax())-1).min()*100,
        'fees_pct': fees_total/10_000*100,
        'n_trades': n_trades,
        'equity': equity,
    }

# === Q1: Cron-tijdstip — daily resample op verschillende uren ===
print("="*100)
print("Q1: Optimaal cron-tijdstip — daily close op uur X UTC")
print("="*100)
print("Test: zelfde MA50-strategie, maar 'daily close' geëxtraheerd op verschillende uren")
print(f"{'Cron uur':<12}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>9}{'#Trades':>10}{'Final€':>10}")
print("-"*100)

df_1h = load_1h()
cron_results = []
for hour in [0, 4, 8, 12, 16, 20, 22]:
    # Filter naar uren waar de bot zou draaien
    hourly_at_h = df_1h[df_1h.index.hour == hour].copy()
    if len(hourly_at_h) < 100:
        print(f"  {hour:02d}:00 UTC: SKIP (te weinig data)")
        continue
    
    r = backtest_trend(hourly_at_h['close'], ma_window=50)
    r['cron_hour_utc'] = hour
    cron_results.append({k:v for k,v in r.items() if k != 'equity'})
    print(f"  {hour:02d}:00 UTC   {r['sharpe']:+10.2f}{r['ann_return_pct']:>+9.1f}%"
          f"{r['mdd_pct']:>+7.1f}%{r['fees_pct']:>7.1f}%{r['n_trades']:>10}{r['final']:>10.0f}")

# === Q2a: DOW skip-strategie ===
print()
print("="*100)
print("Q2a: Skip-DOW strategieën — gebruiken HAR-RS-DOW's weekend-inzicht")
print("="*100)
print("HAR-RS-DOW vond: Zaterdag γ = -0.281 (laagste vol), Maandag γ = +0.142 (hoogste)")
print("Hypothese: dagen met lage vol = wijde spreads = duurdere uitvoering. Mijden helpt?")
print()

daily_22 = df_1h[df_1h.index.hour == 22]['close']  # 22:00 UTC = jouw huidige cron-tijd

print(f"{'Strategie':<45}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>9}{'#Trades':>10}{'Final€':>10}")
print("-"*100)

# Baseline
r0 = backtest_trend(daily_22, ma_window=50)
print(f"  {'Baseline: trade alle dagen':<43}{r0['sharpe']:+10.2f}{r0['ann_return_pct']:>+9.1f}%"
      f"{r0['mdd_pct']:>+7.1f}%{r0['fees_pct']:>7.1f}%{r0['n_trades']:>10}{r0['final']:>10.0f}")

# Skip Saturday (dow=5)
r_satskip = backtest_trend(daily_22, day_filter=lambda t: t.dayofweek != 5, ma_window=50)
print(f"  {'Skip Saturday (low-vol dag)':<43}{r_satskip['sharpe']:+10.2f}"
      f"{r_satskip['ann_return_pct']:>+9.1f}%{r_satskip['mdd_pct']:>+7.1f}%"
      f"{r_satskip['fees_pct']:>7.1f}%{r_satskip['n_trades']:>10}{r_satskip['final']:>10.0f}")

# Skip weekend (dow=5,6)
r_wkskip = backtest_trend(daily_22, day_filter=lambda t: t.dayofweek not in (5, 6), ma_window=50)
print(f"  {'Skip weekend (Sat + Sun)':<43}{r_wkskip['sharpe']:+10.2f}"
      f"{r_wkskip['ann_return_pct']:>+9.1f}%{r_wkskip['mdd_pct']:>+7.1f}%"
      f"{r_wkskip['fees_pct']:>7.1f}%{r_wkskip['n_trades']:>10}{r_wkskip['final']:>10.0f}")

# Only trade on Monday (highest vol)
r_monly = backtest_trend(daily_22, day_filter=lambda t: t.dayofweek == 0, ma_window=50)
print(f"  {'Only Monday (highest vol DOW)':<43}{r_monly['sharpe']:+10.2f}"
      f"{r_monly['ann_return_pct']:>+9.1f}%{r_monly['mdd_pct']:>+7.1f}%"
      f"{r_monly['fees_pct']:>7.1f}%{r_monly['n_trades']:>10}{r_monly['final']:>10.0f}")

# === Q2b: DOW-aware position sizing ===
print()
print("="*100)
print("Q2b: DOW-aware position sizing — schaal positie op basis van verwachte vol")
print("="*100)
print("Idee: HAR-RS-DOW vond systematische DOW vol-effecten. Schaal positie inverse:")
print("  - Maandag (high vol):  smaller positie (target_w × 0.8)")
print("  - Zaterdag (low vol):  larger positie (target_w × 1.0, capped 0.90)")
print()

dow_scalers = {0: 0.80, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.00, 6: 0.95}
def dow_scale(t):
    return dow_scalers.get(t.dayofweek, 1.0)

r_dow = backtest_trend(daily_22, position_scaler=dow_scale, ma_window=50)
print(f"{'Strategie':<45}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Final€':>10}")
print("-"*100)
print(f"  {'Baseline (geen DOW-scaling)':<43}{r0['sharpe']:+10.2f}"
      f"{r0['ann_return_pct']:>+9.1f}%{r0['mdd_pct']:>+7.1f}%{r0['final']:>10.0f}")
print(f"  {'DOW-aware sizing (per HAR-RS-DOW γ)':<43}{r_dow['sharpe']:+10.2f}"
      f"{r_dow['ann_return_pct']:>+9.1f}%{r_dow['mdd_pct']:>+7.1f}%{r_dow['final']:>10.0f}")

# === Save & plot ===
all_results = {
    'cron_hour': pd.DataFrame(cron_results),
    'dow_filter': pd.DataFrame([
        {'name': 'baseline', **{k:v for k,v in r0.items() if k != 'equity'}},
        {'name': 'skip_saturday', **{k:v for k,v in r_satskip.items() if k != 'equity'}},
        {'name': 'skip_weekend', **{k:v for k,v in r_wkskip.items() if k != 'equity'}},
        {'name': 'only_monday', **{k:v for k,v in r_monly.items() if k != 'equity'}},
    ]),
}
all_results['cron_hour'].to_csv(P/'outputs/tables/H10_cron_hour.csv', index=False)
all_results['dow_filter'].to_csv(P/'outputs/tables/H10_dow_filter.csv', index=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sharpe per cron-uur
ax = axes[0, 0]
hrs = [r['cron_hour_utc'] for r in cron_results]
shrs = [r['sharpe'] for r in cron_results]
ax.bar(hrs, shrs, color='steelblue', alpha=0.8)
for i, (h, s) in enumerate(zip(hrs, shrs)):
    ax.text(h, s + 0.02, f'{s:+.2f}', ha='center', fontsize=9)
ax.set_xlabel('Cron-uur UTC'); ax.set_ylabel('Sharpe ratio')
ax.set_title('Q1: Sharpe per cron-uur (Pure Trend MA50)', fontweight='bold')
ax.axvline(22, color='red', ls=':', alpha=0.5, label='Huidige cron-tijd (22:05 UTC)')
ax.legend(); ax.grid(True, alpha=0.3)

# Final wealth per cron-uur
ax = axes[0, 1]
finals = [r['final'] for r in cron_results]
ax.bar(hrs, finals, color='darkgreen', alpha=0.8)
ax.axhline(10_000, color='black', ls='--', lw=0.5, label='Starting capital')
ax.set_xlabel('Cron-uur UTC'); ax.set_ylabel('Final wealth (€)')
ax.set_title('Q1: Final wealth per cron-uur', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# DOW filter comparison
ax = axes[1, 0]
dow_df = all_results['dow_filter']
ax.barh(dow_df['name'], dow_df['sharpe'], color='purple', alpha=0.8)
for i, (name, sh) in enumerate(zip(dow_df['name'], dow_df['sharpe'])):
    ax.text(sh + 0.02, i, f'{sh:+.2f}', va='center', fontsize=9)
ax.set_xlabel('Sharpe ratio')
ax.set_title('Q2a: DOW-skip strategieën', fontweight='bold')
ax.axvline(0, color='black', lw=0.5); ax.grid(True, alpha=0.3)

# DOW vol effect from HAR
ax = axes[1, 1]
dow_gamma = {'Mon': 0.142, 'Tue': 0.087, 'Wed': 0.063, 'Thu': 0.054, 
              'Fri': 0.038, 'Sat': -0.281, 'Sun': 0.0}
days = list(dow_gamma.keys())
gammas = list(dow_gamma.values())
colors = ['red' if g > 0 else 'blue' if g < 0 else 'gray' for g in gammas]
ax.bar(days, gammas, color=colors, alpha=0.8)
ax.set_ylabel('HAR-RS-DOW γ coefficient')
ax.set_title('Reference: DOW vol coefficient uit HAR-RS-DOW', fontweight='bold')
ax.axhline(0, color='black', lw=0.5); ax.grid(True, alpha=0.3)

fig.suptitle('H10 — Cron-timing + DOW-filtering test voor Pure Trend MA50',
              fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H10_timing_dow_test.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Saved: outputs/tables/H10_*.csv")
print(f"✓ Saved: outputs/figures/H10_timing_dow_test.png")
