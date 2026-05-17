"""H9 — Frequentie-test voor Pure Trend MA50.

Antwoordt op: 'Moet de bot daily, 1h, 5m, of 1m draaien?'

Approach A: Daily MA50 signal, verschillende rebalance-frequenties
  → Test of vaker draaien helpt (zelfde signal, ander cadence)
  
Approach B: MA50 op natuurlijke frequentie (50 bars)
  → Test of meer granulair signaal helpt (ander signal, ander cadence)

OOS window: 2022-10-01 → 2026-05-14
Fee model: Bitvavo maker 15bps (huidig)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# OOS boundaries
OOS_START = pd.Timestamp('2022-10-01')
OOS_END = pd.Timestamp('2026-05-14')

def load_candles(freq: str):
    """Load BTC-EUR candles bij gegeven frequentie, snijd tot OOS."""
    df = pd.read_parquet(P / f'data/raw/candles_BTC-EUR_{freq}.parquet')
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('ts')[['close']].sort_index()
    df = df.loc[OOS_START:OOS_END]
    return df

def backtest_trend(prices: pd.Series, ma_window: int, 
                   fee: float = 0.0015, deadband: float = 0.05,
                   max_alloc: float = 0.90):
    """Pure trend backtest: target_w = max_alloc als prijs > MA, anders 0."""
    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    in_trend = (prices > ma).astype(int)
    
    # State
    cash = 10_000.0
    btc = 0.0
    equity = pd.Series(index=prices.index, dtype=float)
    n_trades = 0
    fees_total = 0.0
    current_w = 0.0
    
    for i, t in enumerate(prices.index):
        p = prices.loc[t]
        wealth = cash + btc * p
        equity.loc[t] = wealth
        
        if pd.isna(in_trend.loc[t]):
            continue
        
        target_w = max_alloc * in_trend.loc[t]
        current_w = (btc * p) / wealth if wealth > 0 else 0
        
        # Trade alleen bij significante weight-deviation
        if abs(target_w - current_w) > deadband and i > 0:
            target_btc_val = wealth * target_w
            delta = target_btc_val - btc * p
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
    log_ret = np.log(equity / equity.shift(1)).dropna()
    
    n_yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    final = equity.iloc[-1]
    
    # Annualisatie-factor afhankelijk van frequency van data
    # Inferenced uit gemiddelde tijdsverschil tussen rijen
    avg_dt_sec = pd.Series(equity.index).diff().dt.total_seconds().mean()
    bars_per_year = 365.25 * 24 * 3600 / avg_dt_sec if avg_dt_sec > 0 else 365.25
    
    return {
        'final': final,
        'ann_return_pct': ((final/10_000)**(1/n_yrs) - 1) * 100 if n_yrs > 0 else 0,
        'sharpe_ann': (log_ret.mean() * bars_per_year) / (log_ret.std() * np.sqrt(bars_per_year)) if log_ret.std() > 0 else 0,
        'mdd_pct': ((equity / equity.cummax()) - 1).min() * 100,
        'fees_pct': (fees_total / 10_000) * 100,
        'n_trades': n_trades,
        'equity': equity,
    }


# ===== APPROACH A: zelfde signal (50-daags MA), verschillende rebalance-frequencies =====
print("="*100)
print("APPROACH A: Daily MA50 signal, gerebalanced op {1d, 4h, 1h, 30m, 15m, 5m, 1m}")
print("="*100)
print(f"{'Freq':<8}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>9}{'#Trades':>10}{'Final€':>10}")
print("-"*100)

# Eerst daily MA50 berekenen
daily = load_candles('1d')
daily_ma50 = daily['close'].rolling(50, min_periods=50).mean()
daily_in_trend = (daily['close'] > daily_ma50)

results_a = []

# Voor elke evaluation-frequency: gebruik de daily MA50 (forward-filled naar die freq)
for freq_label, freq_str, resample_rule in [
    ('1d', '1d', 'D'),
    ('4h', '1h', '4h'),
    ('1h', '1h', 'h'),
    ('30m', '15m', '30min'),
    ('15m', '15m', '15min'),
    ('5m', '5m', '5min'),
    ('1m', '1m', 'min'),
]:
    try:
        prices = load_candles(freq_str)['close']
        if resample_rule != freq_str:
            prices = prices.resample(resample_rule).last().dropna()
        
        # Map daily MA50 + in_trend naar deze frequentie via forward-fill
        ma_aligned = daily_ma50.reindex(prices.index, method='ffill')
        # Build synthetic "as if MA50_daily was the signal" — bot decides at this cadence
        # Equivalent: prices vs daily_ma50_ffilled
        
        # Run backtest met deze gebridgde signal
        # Use a placeholder ma_window=1 trick: we already have ma_aligned, so just compare
        cash, btc = 10_000.0, 0.0
        equity = pd.Series(index=prices.index, dtype=float)
        n_trades, fees_total = 0, 0.0
        
        for i, t in enumerate(prices.index):
            p = prices.iloc[i]
            wealth = cash + btc * p
            equity.iloc[i] = wealth
            
            if pd.isna(ma_aligned.iloc[i]):
                continue
            
            in_trend = 1 if p > ma_aligned.iloc[i] else 0
            target_w = 0.90 * in_trend
            current_w = (btc * p) / wealth if wealth > 0 else 0
            
            if abs(target_w - current_w) > 0.05 and i > 0:
                delta = wealth * target_w - btc * p
                fee_eur = abs(delta) * 0.0015
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
        log_ret = np.log(equity / equity.shift(1)).dropna()
        n_yrs = (equity.index[-1] - equity.index[0]).days / 365.25
        avg_dt = pd.Series(equity.index).diff().dt.total_seconds().mean()
        bpy = 365.25 * 24 * 3600 / avg_dt if avg_dt > 0 else 365.25
        sharpe = (log_ret.mean() * bpy) / (log_ret.std() * np.sqrt(bpy)) if log_ret.std() > 0 else 0
        
        r = {
            'freq': freq_label,
            'sharpe': sharpe,
            'ann_return_pct': ((equity.iloc[-1]/10_000)**(1/n_yrs)-1)*100,
            'mdd_pct': ((equity/equity.cummax())-1).min()*100,
            'fees_pct': fees_total/10_000*100,
            'n_trades': n_trades,
            'final': equity.iloc[-1],
            'equity': equity,
        }
        results_a.append(r)
        print(f"{freq_label:<8}{r['sharpe']:+10.2f}{r['ann_return_pct']:>+9.1f}%"
              f"{r['mdd_pct']:>+7.1f}%{r['fees_pct']:>7.1f}%{r['n_trades']:>10}{r['final']:>10.0f}")
    except Exception as e:
        print(f"{freq_label}: FAIL — {e}")

# ===== APPROACH B: natural-frequency MA50 (50 bars per frequency) =====
print()
print("="*100)
print("APPROACH B: MA50 op natuurlijke frequentie (= 50 bars) — fijner signaal bij hogere freq")
print("="*100)
print(f"{'Freq':<8}{'MA period':<14}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>9}{'#Trades':>10}{'Final€':>10}")
print("-"*100)

results_b = []
for freq_label, freq_str, resample_rule, ma_period_label in [
    ('1d',  '1d',  None,     '50 dagen'),
    ('4h',  '1h',  '4h',     '~8 dagen'),
    ('1h',  '1h',  None,     '~2 dagen'),
    ('30m', '15m', '30min',  '~25 uur'),
    ('15m', '15m', None,     '~12.5 uur'),
    ('5m',  '5m',  None,     '~4 uur'),
    ('1m',  '1m',  None,     '50 minuten'),
]:
    try:
        prices = load_candles(freq_str)['close']
        if resample_rule:
            prices = prices.resample(resample_rule).last().dropna()
        
        r = backtest_trend(prices, ma_window=50, fee=0.0015, deadband=0.05)
        r['freq'] = freq_label
        r['ma_label'] = ma_period_label
        results_b.append(r)
        print(f"{freq_label:<8}{ma_period_label:<14}{r['sharpe_ann']:+10.2f}{r['ann_return_pct']:>+9.1f}%"
              f"{r['mdd_pct']:>+7.1f}%{r['fees_pct']:>7.1f}%{r['n_trades']:>10}{r['final']:>10.0f}")
    except Exception as e:
        print(f"{freq_label}: FAIL — {e}")

# === Save & plot ===
pd.DataFrame([{k:v for k,v in r.items() if k != 'equity'} for r in results_a]).to_csv(
    P/'outputs/tables/H9_approach_A_same_signal.csv', index=False)
pd.DataFrame([{k:v for k,v in r.items() if k != 'equity'} for r in results_b]).to_csv(
    P/'outputs/tables/H9_approach_B_natural_freq.csv', index=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A: Sharpe per freq
ax = axes[0, 0]
labels = [r['freq'] for r in results_a]
sharpes = [r['sharpe'] for r in results_a]
ax.bar(range(len(labels)), sharpes, color=['darkgreen' if s>0 else 'red' for s in sharpes])
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_title('Approach A: same daily-MA50 signal, variable execution freq', fontweight='bold')
ax.set_ylabel('Annualized Sharpe')
ax.axhline(0, color='black', lw=0.5); ax.grid(True, alpha=0.3)

# A: Fees per freq
ax = axes[0, 1]
ax.bar(range(len(labels)), [r['fees_pct'] for r in results_a], color='orange')
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_title('Approach A: fees % per execution freq', fontweight='bold')
ax.set_ylabel('Fees as % of initial capital')
ax.grid(True, alpha=0.3)

# B: Sharpe per freq
ax = axes[1, 0]
labels_b = [r['freq'] for r in results_b]
sharpes_b = [r['sharpe_ann'] for r in results_b]
ax.bar(range(len(labels_b)), sharpes_b, color=['darkgreen' if s>0 else 'red' for s in sharpes_b])
ax.set_xticks(range(len(labels_b))); ax.set_xticklabels(labels_b)
ax.set_title('Approach B: natural-frequency MA50 (50 bars per freq)', fontweight='bold')
ax.set_ylabel('Annualized Sharpe')
ax.axhline(0, color='black', lw=0.5); ax.grid(True, alpha=0.3)

# B: Final wealth comparison
ax = axes[1, 1]
ax.bar(range(len(labels_b)), [r['final'] for r in results_b], color='steelblue')
ax.set_xticks(range(len(labels_b))); ax.set_xticklabels(labels_b)
ax.axhline(10_000, color='black', ls='--', lw=0.5, label='Starting capital (€10k)')
ax.set_title('Approach B: final wealth per freq (€10k initial)', fontweight='bold')
ax.set_ylabel('Final wealth (€)'); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle('H9 — Frequentie-test voor Pure Trend MA50: helpt vaker draaien?',
              fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H9_frequency_test.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Saved: outputs/tables/H9_approach_*.csv")
print(f"✓ Saved: outputs/figures/H9_frequency_test.png")
