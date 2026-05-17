"""H11 — HAR-RS-DOW-derived execution filters op pure trend strategy.

Op baseline (pure trend MA50 + skip weekend), test of HAR-outputs extra
trade-filters bieden die schade voorkomen zonder return op te offeren.

Vier filters geïnspireerd op HAR-RS-DOW outputs:
  A. Hoge-vol regime gate    — skip wanneer μ_pred > 90th percentile (turbulence)
  B. Multi-horizon mismatch  — skip wanneer h=1 en h=5 forecasts sterk botsen
  C. Semivariance asymmetry  — skip wanneer recent RS-/RS+ > 2 (downside dominant)
  D. Post-jump filter        — skip dag NA een Barndorff-Nielsen jump (J_flag)

Evaluatie: zowel full OOS (2022-2026) als stress-period (2024-2026) om
multiple-testing/overfitting te detecteren.
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

# === Load all data sources ===
# 1. E1 walkforward (h=1 HAR forecasts, density params)
df_e1 = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df_e1.index = pd.to_datetime(df_e1.index)
if df_e1.index.tz is not None: df_e1.index = df_e1.index.tz_localize(None)

# 2. E4b multi-horizon (h=1 and h=5 forecasts)
df_e4 = pd.read_parquet(P/'outputs/tables/E4b_A_perday_HAR-RS-DOW.parquet')
df_e4.index = pd.to_datetime(df_e4.index)
if df_e4.index.tz is not None: df_e4.index = df_e4.index.tz_localize(None)

# 3. Realized measures (RS+, RS-, jumps)
df_rm = pd.read_parquet(P/'data/processed/realized_measures_btc_eur_v2.parquet')
df_rm['date'] = pd.to_datetime(df_rm['date'])
df_rm = df_rm.set_index('date')
if df_rm.index.tz is not None:
    df_rm.index = df_rm.index.tz_localize(None)

# 4. Daily prices (1h candles, 22:00 UTC closes — matching huidige cron)
candles_1h = pd.read_parquet(P/'data/raw/candles_BTC-EUR_1h.parquet')
candles_1h['ts'] = pd.to_datetime(candles_1h['ts'], unit='ms', utc=True).dt.tz_localize(None)
candles_1h = candles_1h.set_index('ts').sort_index()
daily_22 = candles_1h[candles_1h.index.hour == 22]['close'].copy()
daily_22.index = daily_22.index.normalize()

# Join data sources op dag-niveau
data = pd.DataFrame({'close': daily_22})
data = data.join(df_e1[['mu_pred', 'sigma_pred']], how='left')
data = data.join(df_e4[['mu_h1', 'mu_h5']], how='left')
data = data.join(df_rm[['RS_pos', 'RS_neg', 'J_flag', 'Z']], how='left')
data = data.loc[OOS_START:OOS_END].dropna(subset=['close', 'mu_pred'])

print(f"Joined data: {len(data)} dagen, {data.index[0].date()} → {data.index[-1].date()}")

# === Compute filter signals (rolling-window threshold to avoid lookahead) ===
W = 60  # rolling window voor empirical thresholds

# A: hoge-vol regime (μ_pred > 90th percentile in trailing window)
data['mu_p90'] = data['mu_pred'].rolling(W, min_periods=20).quantile(0.90).shift(1)
data['filter_A_skip'] = (data['mu_pred'] > data['mu_p90']).astype(int).fillna(0)

# B: multi-horizon mismatch (|μ_h5 - μ_h1| > 90th percentile)
data['horizon_gap'] = (data['mu_h5'] - data['mu_h1']).abs()
data['gap_p90'] = data['horizon_gap'].rolling(W, min_periods=20).quantile(0.90).shift(1)
data['filter_B_skip'] = (data['horizon_gap'] > data['gap_p90']).astype(int).fillna(0)

# C: semivariance asymmetry — RS-/RS+ > 2x (lag voor causal use)
data['rs_ratio'] = (data['RS_neg'] / data['RS_pos'].clip(lower=1e-10)).shift(1)
data['filter_C_skip'] = (data['rs_ratio'] > 2.0).astype(int).fillna(0)

# D: post-jump filter — J_flag was True gisteren
data['filter_D_skip'] = data['J_flag'].shift(1).fillna(0).astype(int)

# Skip-weekend baseline filter
data['filter_weekend_skip'] = data.index.dayofweek.isin([5, 6]).astype(int)

print(f"\nFilter trigger frequencies (% of days):")
for f in ['filter_weekend_skip', 'filter_A_skip', 'filter_B_skip', 'filter_C_skip', 'filter_D_skip']:
    pct = data[f].mean() * 100
    print(f"  {f:<30}: {pct:5.1f}%")

# === Backtest engine met filter-stack ===
def backtest(data, filters_to_skip=None, ma_window=50, fee=0.0015, deadband=0.05):
    """Pure trend MA50 met optionele extra skip-filters."""
    if filters_to_skip is None: filters_to_skip = ['filter_weekend_skip']
    
    prices = data['close']
    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    
    cash, btc = 10_000.0, 0.0
    equity = pd.Series(index=prices.index, dtype=float)
    n_trades, n_skips, fees_total = 0, 0, 0.0
    
    for i, t in enumerate(prices.index):
        p = prices.iloc[i]
        wealth = cash + btc * p
        equity.iloc[i] = wealth
        
        if pd.isna(ma.iloc[i]):
            continue
        
        # Active filters: skip if ANY filter is on
        skip_today = any(data[f].iloc[i] == 1 for f in filters_to_skip)
        if skip_today:
            n_skips += 1
            continue
        
        in_trend = 1 if p > ma.iloc[i] else 0
        target_w = 0.90 * in_trend
        current_w = (btc * p) / wealth if wealth > 0 else 0
        
        if abs(target_w - current_w) > deadband and i > 0:
            delta = wealth * target_w - btc * p
            fee_eur = abs(delta) * fee
            if delta > 0:
                btc += (delta - fee_eur) / p; cash -= delta
            else:
                btc += delta / p; cash -= delta - fee_eur
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
        'n_skips': n_skips,
        'equity': equity,
    }

# === Test individuele filters bovenop baseline (weekend skip) ===
def run_section(data_slice, label):
    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    print(f"Sample: {data_slice.index[0].date()} → {data_slice.index[-1].date()} (n={len(data_slice)})")
    print()
    print(f"{'Filter combinatie':<48}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>8}{'#Trades':>9}{'#Skips':>9}")
    print('-'*100)
    
    tests = [
        ('Baseline (weekend skip only)',                    ['filter_weekend_skip']),
        ('Baseline + A (high-vol regime gate)',             ['filter_weekend_skip', 'filter_A_skip']),
        ('Baseline + B (multi-horizon mismatch)',           ['filter_weekend_skip', 'filter_B_skip']),
        ('Baseline + C (semivariance asymmetry)',           ['filter_weekend_skip', 'filter_C_skip']),
        ('Baseline + D (post-jump filter)',                 ['filter_weekend_skip', 'filter_D_skip']),
        ('Baseline + A + B (vol + horizon)',                ['filter_weekend_skip', 'filter_A_skip', 'filter_B_skip']),
        ('Baseline + A + C (vol + asymmetry)',              ['filter_weekend_skip', 'filter_A_skip', 'filter_C_skip']),
        ('Baseline + A + D (vol + post-jump)',              ['filter_weekend_skip', 'filter_A_skip', 'filter_D_skip']),
        ('Baseline + ALLE filters',                         ['filter_weekend_skip', 'filter_A_skip', 'filter_B_skip', 'filter_C_skip', 'filter_D_skip']),
    ]
    results = []
    for name, filters in tests:
        r = backtest(data_slice, filters_to_skip=filters)
        r['name'] = name
        r['n_filters'] = len(filters)
        results.append(r)
        marker = ''
        # Highlight best Sharpe
        marker = ' ⭐' if r['sharpe'] == max(t.get('sharpe', 0) for t in results) else ''
        print(f"  {name:<46}{r['sharpe']:+10.2f}{r['ann_return_pct']:>+9.1f}%"
              f"{r['mdd_pct']:>+7.1f}%{r['fees_pct']:>7.1f}%{r['n_trades']:>9}{r['n_skips']:>9}{marker}")
    return results

results_full = run_section(data, "FULL OOS (2022-10-01 → 2026-05-14)")
results_stress = run_section(data.loc['2024-01-01':], "STRESS PERIOD (2024-01-01 → 2026-05-14)")

# === Combineer: welke filter werkt in BEIDE periodes? ===
print(f"\n{'='*100}")
print(f"OVERFITTING CHECK: welke filter werkt in zowel full als stress?")
print(f"{'='*100}")
print(f"{'Filter':<48}{'Full Sharpe':>14}{'Stress Sharpe':>16}{'Verdict':>22}")
print('-'*100)

baseline_full = results_full[0]['sharpe']
baseline_stress = results_stress[0]['sharpe']
print(f"  {'Baseline (weekend only)':<46}{baseline_full:+14.2f}{baseline_stress:+16.2f}{'(reference)':>22}")

for r_f, r_s in zip(results_full[1:], results_stress[1:]):
    improves_full = r_f['sharpe'] > baseline_full + 0.02
    improves_stress = r_s['sharpe'] > baseline_stress + 0.02
    if improves_full and improves_stress:
        verdict = '✓ ROBUST'
    elif improves_full and not improves_stress:
        verdict = '⚠ in-sample only'
    elif not improves_full and improves_stress:
        verdict = '? stress only'
    else:
        verdict = '✗ no improvement'
    print(f"  {r_f['name']:<46}{r_f['sharpe']:+14.2f}{r_s['sharpe']:+16.2f}{verdict:>22}")

# Save & plot
pd.DataFrame([{k:v for k,v in r.items() if k != 'equity'} 
              for r in results_full]).to_csv(P/'outputs/tables/H11_filters_full.csv', index=False)
pd.DataFrame([{k:v for k,v in r.items() if k != 'equity'} 
              for r in results_stress]).to_csv(P/'outputs/tables/H11_filters_stress.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top: Sharpe per filter — full
ax = axes[0, 0]
names = [r['name'].replace('Baseline + ', '').replace('Baseline ', '') for r in results_full]
shrs = [r['sharpe'] for r in results_full]
colors = ['#2ecc71' if s > baseline_full else '#e74c3c' if s < baseline_full else 'gray' for s in shrs]
ax.barh(names, shrs, color=colors, alpha=0.8)
ax.axvline(baseline_full, color='blue', ls='--', label=f'Baseline ({baseline_full:+.2f})')
ax.set_xlabel('Sharpe ratio'); ax.set_title('Full OOS — Filter impact', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='x')

# Top right: Sharpe per filter — stress
ax = axes[0, 1]
shrs_s = [r['sharpe'] for r in results_stress]
colors_s = ['#2ecc71' if s > baseline_stress else '#e74c3c' if s < baseline_stress else 'gray' for s in shrs_s]
ax.barh(names, shrs_s, color=colors_s, alpha=0.8)
ax.axvline(baseline_stress, color='blue', ls='--', label=f'Baseline ({baseline_stress:+.2f})')
ax.set_xlabel('Sharpe ratio'); ax.set_title('Stress 2024-2026 — Filter impact', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='x')

# Bottom: full equity curves
ax = axes[1, 0]
for r in results_full[:5]:  # top 5 om plot leesbaar te houden
    eq_norm = r['equity'] / 10_000
    ax.plot(eq_norm.index, eq_norm.values, lw=1.2, label=f"{r['name'][:30]} (Sh={r['sharpe']:+.2f})")
ax.axhline(1, color='gray', ls='--', lw=0.5)
ax.set_title('Equity curves — top filters full OOS', fontweight='bold')
ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.3)

# Bottom right: filter trigger frequencies
ax = axes[1, 1]
trigger_pcts = {
    'Weekend': data['filter_weekend_skip'].mean()*100,
    'A: high vol': data['filter_A_skip'].mean()*100,
    'B: h1≠h5': data['filter_B_skip'].mean()*100,
    'C: RS-/RS+': data['filter_C_skip'].mean()*100,
    'D: post-jump': data['filter_D_skip'].mean()*100,
}
ax.bar(trigger_pcts.keys(), trigger_pcts.values(), color='orange', alpha=0.8)
for i, (k, v) in enumerate(trigger_pcts.items()):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
ax.set_ylabel('% of days filter triggers')
ax.set_title('Filter trigger frequency', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('H11 — Welke HAR-RS-DOW-derived execution filters helpen?',
              fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H11_har_filters.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"\n✓ Saved: outputs/tables/H11_filters_*.csv, outputs/figures/H11_har_filters.png")
