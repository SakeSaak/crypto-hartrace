"""H8 — Robustheids-extensies op de H7 scenario-vergelijking.

1. MA-length robustness  : test trend met windows 10/20/50/100/200
2. Covered-call yield    : vary premium yield 0%-20% — wanneer breakt scenario C?
3. Stress-test 2024-2026 : alle strategieën in de bear-achtige sub-periode
4. Walk-forward selection: kies elke 3 maanden de beste strategie uit 6m trailing
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# Load data
df_har = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df_har.index = pd.to_datetime(df_har.index)
if df_har.index.tz is not None: df_har.index = df_har.index.tz_localize(None)
df_har['sigma_har'] = np.sqrt(np.exp(df_har['mu_pred'] + 0.5 * df_har['sigma_pred']**2))
df_har['sigma_har_ann'] = df_har['sigma_har'] * np.sqrt(365)
df_har['S'] = 100 * (1 + df_har['r_actual']).cumprod()

def true_range(df):
    return pd.concat([df['S'].diff().abs(), df['S'] - df['S'].shift()], axis=1).abs().max(axis=1)
df_har['atr14'] = true_range(df_har).rolling(14).mean()

# === Backtest engine (klein, hergebruikbaar) ===
def run_strategy(strategy, data, initial=1000, fee_maker=0.0015, target_vol=0.35,
                  ma_window=50, atr_mult=3.0, cc_yield=0.0, rebal_threshold=0.02):
    """Schoon backtest-functie die werkt op ELKE data-slice."""
    cash, btc = initial, 0.0
    fees_total = 0.0
    n_trades = 0
    equity = pd.Series(index=data.index, dtype=float)
    
    in_pos = False
    current_stop = -np.inf
    
    # Pre-compute MA
    ma = data['S'].rolling(ma_window, min_periods=ma_window).mean()
    in_trend = (data['S'] > ma).astype(int)
    
    for i, t in enumerate(data.index):
        row = data.loc[t]
        price = row['S']
        wealth = cash + btc * price
        equity.loc[t] = wealth
        
        if pd.isna(in_trend.loc[t]) or pd.isna(row.get('sigma_har_ann', np.nan)):
            continue
        
        if strategy == 'buy_and_hold':
            target_w = 1.0
        elif strategy == 'pure_trend':
            target_w = float(in_trend.loc[t])
        elif strategy == 'har_voltarget_trend':
            target_w = min(1.0, target_vol / row['sigma_har_ann']) if in_trend.loc[t] else 0
        elif strategy == 'atr_stop_trend':
            if pd.isna(row['atr14']) or row['atr14'] == 0:
                target_w = 0
            elif not in_pos and in_trend.loc[t]:
                target_w, current_stop, in_pos = 0.90, price - atr_mult*row['atr14'], True
            elif in_pos:
                current_stop = max(current_stop, price - atr_mult*row['atr14'])
                if price < current_stop or not in_trend.loc[t]:
                    target_w, in_pos = 0, False
                else:
                    target_w = 0.90
            else:
                target_w = 0
        elif strategy == 'covered_call_overlay':
            target_w = 0.90 if in_trend.loc[t] else 0.30
        else:
            raise ValueError(strategy)
        
        if i > 0:
            target_btc_value = wealth * target_w
            delta = target_btc_value - btc * price
            if abs(delta) / wealth > rebal_threshold:
                fee = abs(delta) * fee_maker
                if delta > 0:
                    buyable = (delta - fee) / price
                    btc += buyable; cash -= delta
                else:
                    btc += delta / price; cash += (-delta - fee)
                cash -= fee
                fees_total += fee
                n_trades += 1
            
            # CC premium yield (theoretical) - daily compound
            if strategy == 'covered_call_overlay' and cc_yield > 0 and btc > 0:
                cash += (cc_yield / 365) * btc * price
    
    equity = equity.dropna()
    log_ret = np.log(equity / equity.shift(1)).dropna()
    final = equity.iloc[-1]
    n_yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    
    return {
        'final': final, 'initial': initial,
        'ann_return': ((final/initial)**(1/n_yrs)-1)*100 if n_yrs > 0 else 0,
        'sharpe': (log_ret.mean()*365)/(log_ret.std()*np.sqrt(365)) if log_ret.std() > 0 else 0,
        'mdd_pct': ((equity/equity.cummax())-1).min()*100,
        'fees_pct': fees_total/initial*100,
        'n_trades': n_trades,
        'equity': equity,
    }

# ===== ANALYSE 1: MA-LENGTH ROBUSTNESS =====
print("="*100)
print("ANALYSE 1: MA-length robustness — is MA50 toeval of principe?")
print("="*100)
print(f"{'MA window':<12}{'Strategy':<28}{'Sharpe':>8}{'AnnRet%':>10}{'MDD%':>8}{'Fees%':>8}{'#Trades':>8}")
print("-"*100)

ma_results = []
for ma_w in [10, 20, 50, 100, 200]:
    for strat in ['pure_trend', 'atr_stop_trend', 'har_voltarget_trend']:
        r = run_strategy(strat, df_har, initial=5000, ma_window=ma_w)
        ma_results.append({**{k:v for k,v in r.items() if k!='equity'}, 'ma_window': ma_w, 'strategy': strat})
        print(f"  MA{ma_w:<8}{strat:<28}{r['sharpe']:+8.2f}{r['ann_return']:>+9.1f}%"
              f"{r['mdd_pct']:>+7.1f}%{r['fees_pct']:>7.1f}%{r['n_trades']:>8}")

# Best MA per strategy
print("\nBeste MA-lengte per strategie (op Sharpe):")
df_ma = pd.DataFrame(ma_results)
for strat in ['pure_trend', 'atr_stop_trend', 'har_voltarget_trend']:
    best = df_ma[df_ma['strategy']==strat].nlargest(1, 'sharpe').iloc[0]
    print(f"  {strat:<28} → MA{int(best['ma_window']):<3}, Sharpe={best['sharpe']:+.2f}")

# ===== ANALYSE 2: COVERED-CALL YIELD SENSITIVITY =====
print("\n" + "="*100)
print("ANALYSE 2: Covered-call yield sensitivity — wanneer wint Scenario C?")
print("="*100)
print(f"{'Premium yield':<18}{'Sharpe':>10}{'AnnRet%':>10}{'MDD%':>8}{'vs Pure Trend':>16}")
print("-"*100)

baseline = run_strategy('pure_trend', df_har, initial=5000, ma_window=50)
baseline_sharpe = baseline['sharpe']
print(f"  Baseline (Pure Trend MA50): Sharpe = {baseline_sharpe:+.2f}, AnnRet = {baseline['ann_return']:+.1f}%\n")

cc_results = []
for cc_y in [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    r = run_strategy('covered_call_overlay', df_har, initial=5000, cc_yield=cc_y)
    cc_results.append({**{k:v for k,v in r.items() if k!='equity'}, 'cc_yield': cc_y})
    edge = r['sharpe'] - baseline_sharpe
    marker = '⭐ wins' if edge > 0.05 else ('break-even' if abs(edge) < 0.05 else 'loses')
    print(f"  {int(cc_y*100):>3}%/yr        {r['sharpe']:+10.2f}{r['ann_return']:>+9.1f}%"
          f"{r['mdd_pct']:>+7.1f}%   {marker:>10}")

print("\nInterpretatie: bij welk yield-level wordt Scenario C beter dan pure trend?")
for r in cc_results:
    if r['sharpe'] > baseline_sharpe + 0.05:
        print(f"  → Break-even level: covered-call yield ≥ {int(r['cc_yield']*100)}% per jaar")
        break
else:
    print("  → Geen yield-level bevoordeelt scenario C significant")

# ===== ANALYSE 3: STRESS-TEST 2024-2026 (bear-achtige sub-periode) =====
print("\n" + "="*100)
print("ANALYSE 3: Stress-test 2024-01-01 → 2026-05-14 — survives in moeilijker regime?")
print("="*100)

stress_data = df_har.loc['2024-01-01':].copy()
print(f"Sample: {stress_data.index[0].date()} → {stress_data.index[-1].date()} (n={len(stress_data)})")
btc_period_return = (stress_data['S'].iloc[-1] / stress_data['S'].iloc[0] - 1) * 100
print(f"BTC totale return in deze periode: {btc_period_return:+.1f}%")
print()
print(f"{'Strategy':<28}{'Sharpe':>8}{'AnnRet%':>10}{'MDD%':>8}{'Final€':>10}")
print("-"*100)

stress_results = []
for strat in ['buy_and_hold', 'pure_trend', 'har_voltarget_trend', 'atr_stop_trend']:
    r = run_strategy(strat, stress_data, initial=5000, ma_window=50)
    stress_results.append({**{k:v for k,v in r.items() if k!='equity'}, 'strategy': strat})
    print(f"  {strat:<28}{r['sharpe']:+8.2f}{r['ann_return']:>+9.1f}%"
          f"{r['mdd_pct']:>+7.1f}%{r['final']:>10.0f}")

# Also test pure trend at different MAs in stress period
print("\nPure Trend met verschillende MAs in stress periode:")
for ma_w in [20, 50, 100, 200]:
    r = run_strategy('pure_trend', stress_data, initial=5000, ma_window=ma_w)
    print(f"  Pure Trend MA{ma_w:<4}: Sharpe={r['sharpe']:+.2f}, AnnRet={r['ann_return']:+.1f}%, MDD={r['mdd_pct']:+.1f}%")

# ===== ANALYSE 4: WALK-FORWARD STRATEGY SELECTION =====
print("\n" + "="*100)
print("ANALYSE 4: Walk-forward strategie-selectie — kun je adaptief de winnaar kiezen?")
print("="*100)
print("Setup: 6m trailing-Sharpe lookback, 3m hold-period, herselectie elke 3 maanden")
print()

LOOKBACK_DAYS = 180
HOLD_DAYS = 90
candidates = ['buy_and_hold', 'pure_trend', 'har_voltarget_trend', 'atr_stop_trend']

# Voor walk-forward: simuleer voor elke strategie de daily returns, dan rebalance op selection days
# Eerst: compute daily strategy returns ALS we elke strategie volledig had gedraaid
daily_returns = {}
for strat in candidates:
    r = run_strategy(strat, df_har, initial=10000, ma_window=50)
    daily_returns[strat] = np.log(r['equity']/r['equity'].shift(1)).fillna(0)

# Walk-forward: elke 90 dagen, kies de strategie met beste trailing Sharpe
selection_history = []
wf_equity = pd.Series(index=df_har.index, dtype=float)
wf_equity.iloc[0] = 10000

current_strategy = 'buy_and_hold'  # initial fallback
start_idx = LOOKBACK_DAYS  # we need history for first selection
selection_dates = df_har.index[start_idx::HOLD_DAYS]

for i, sel_date in enumerate(selection_dates):
    # Look back 180 days
    lookback_start = sel_date - pd.Timedelta(days=LOOKBACK_DAYS)
    
    # Compute Sharpe for each candidate over the lookback window
    sharpes = {}
    for strat in candidates:
        sr = daily_returns[strat].loc[lookback_start:sel_date]
        if len(sr) > 30 and sr.std() > 0:
            sharpes[strat] = (sr.mean() * 365) / (sr.std() * np.sqrt(365))
        else:
            sharpes[strat] = -999
    
    # Select best
    best = max(sharpes, key=sharpes.get)
    selection_history.append({
        'date': sel_date, 'selected': best, 'trailing_sharpe': sharpes[best],
        **{f'sh_{s}': sharpes[s] for s in candidates}
    })
    current_strategy = best

# Build walk-forward equity by chaining selected strategy returns
wf_returns = []
selection_history_df = pd.DataFrame(selection_history)

current_strat = 'buy_and_hold'
for t in df_har.index:
    # Find which strategy is active op deze dag
    past_selections = selection_history_df[selection_history_df['date'] <= t]
    if not past_selections.empty:
        current_strat = past_selections.iloc[-1]['selected']
    if t in daily_returns[current_strat].index:
        wf_returns.append(daily_returns[current_strat].loc[t])
    else:
        wf_returns.append(0)

wf_log_returns = pd.Series(wf_returns, index=df_har.index)
wf_equity = 10000 * np.exp(wf_log_returns.cumsum())
wf_log_returns_clean = wf_log_returns[wf_log_returns != 0]

wf_metrics = {
    'sharpe': (wf_log_returns_clean.mean()*365)/(wf_log_returns_clean.std()*np.sqrt(365)),
    'final': wf_equity.iloc[-1],
    'ann_return': ((wf_equity.iloc[-1]/10000)**(1/((wf_equity.index[-1]-wf_equity.index[0]).days/365.25))-1)*100,
    'mdd_pct': ((wf_equity/wf_equity.cummax())-1).min()*100,
}

print(f"Walk-forward (adaptive):  Sharpe={wf_metrics['sharpe']:+.2f}, AnnRet={wf_metrics['ann_return']:+.1f}%, "
      f"MDD={wf_metrics['mdd_pct']:+.1f}%, Final=€{wf_metrics['final']:.0f}")
print("\nVergelijking met fixed strategies (€10k initial, zelfde sample):")
for strat in candidates:
    r = run_strategy(strat, df_har, initial=10000, ma_window=50)
    print(f"  Fixed {strat:<22}: Sharpe={r['sharpe']:+.2f}, AnnRet={r['ann_return']:+.1f}%, MDD={r['mdd_pct']:+.1f}%, Final=€{r['final']:.0f}")

print(f"\nFrequentie van strategie-selecties (n={len(selection_history)} selection-dates):")
sel_counts = selection_history_df['selected'].value_counts()
for s, c in sel_counts.items():
    print(f"  {s}: {c} keer geselecteerd ({c/len(selection_history)*100:.0f}%)")

# Save all results
pd.DataFrame(ma_results).to_csv(P/'outputs/tables/H8_ma_length_robustness.csv', index=False)
pd.DataFrame(cc_results).to_csv(P/'outputs/tables/H8_cc_yield_sensitivity.csv', index=False)
pd.DataFrame(stress_results).to_csv(P/'outputs/tables/H8_stress_test_2024_2026.csv', index=False)
selection_history_df.to_csv(P/'outputs/tables/H8_walkforward_selections.csv', index=False)
print(f"\n✓ Resultaten opgeslagen in outputs/tables/H8_*.csv")

# ===== FIGUREN =====
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: MA length sensitivity
ax = axes[0, 0]
df_ma_plot = pd.DataFrame(ma_results)
for strat in ['pure_trend', 'atr_stop_trend', 'har_voltarget_trend']:
    sub = df_ma_plot[df_ma_plot['strategy']==strat].sort_values('ma_window')
    ax.plot(sub['ma_window'], sub['sharpe'], 'o-', label=strat, markersize=8)
ax.set_xlabel('MA window (days)')
ax.set_ylabel('Sharpe ratio')
ax.set_title('1. MA-length robustness', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: CC yield sensitivity
ax = axes[0, 1]
df_cc = pd.DataFrame(cc_results)
ax.plot(df_cc['cc_yield']*100, df_cc['sharpe'], 'o-', color='purple', markersize=8, label='Covered-call overlay')
ax.axhline(baseline_sharpe, color='gray', ls='--', label=f'Pure Trend MA50 baseline (Sharpe={baseline_sharpe:+.2f})')
ax.set_xlabel('Theoretical covered-call yield (% per year)')
ax.set_ylabel('Sharpe ratio')
ax.set_title('2. Covered-call yield sensitivity', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 3: Stress test
ax = axes[1, 0]
for strat in ['buy_and_hold', 'pure_trend', 'har_voltarget_trend', 'atr_stop_trend']:
    r = run_strategy(strat, stress_data, initial=5000, ma_window=50)
    eq_norm = r['equity'] / r['initial']
    ax.plot(eq_norm.index, eq_norm.values, lw=1.5, label=f"{strat} (Sh={r['sharpe']:+.2f})")
ax.axhline(1, color='gray', ls='--', lw=0.5)
ax.set_ylabel('Wealth / Initial')
ax.set_title('3. Stress-test 2024-2026 only', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 4: Walk-forward
ax = axes[1, 1]
wf_norm = wf_equity / 10000
ax.plot(wf_norm.index, wf_norm.values, lw=2, color='red', label=f'Walk-forward (Sh={wf_metrics["sharpe"]:+.2f})')
# Add fixed strategies for comparison
for strat in candidates:
    r = run_strategy(strat, df_har, initial=10000, ma_window=50)
    eq_norm = r['equity'] / 10000
    ax.plot(eq_norm.index, eq_norm.values, lw=1, alpha=0.5, label=f"{strat} (Sh={r['sharpe']:+.2f})")
ax.axhline(1, color='gray', ls='--', lw=0.5)
ax.set_ylabel('Wealth / Initial')
ax.set_title('4. Walk-forward strategy selection', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle('H8 — Vier robustheids-extensies op de scenario-vergelijking',
              fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H8_robustness_extensions.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"✓ Figuur: outputs/figures/H8_robustness_extensions.png")
