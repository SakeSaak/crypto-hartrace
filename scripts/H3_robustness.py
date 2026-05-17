"""H3 — Robustness-scorecard voor alle 11 H2-strategieën.

Drie onafhankelijke checks per strategie:
  1. Sub-period stability: BTC Early (2022-10→2024-07) vs Late (2024-07→2026-05)
  2. Cross-asset transfer: zelfde regel op ETH-EUR (geen HAR voor ETH dus HAR-strats skip)
  3. ATR-multiplier sensitivity: 2.0 / 2.5 / 3.0 / 3.5 / 4.0 (alleen voor ATR-stop)

Verdict per strategie: ROBUST / CONDITIONAL / FRAGILE.
"""
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')

# Import strategy functions van H2
from H2_TA_strategies import (
    s_buyhold, s_trend_ma50, s_bollinger_mr, s_bollinger_bo, s_macd,
    s_atr_stop, s_donchian, s_rsi_bounce, s_har_volregime_trend,
    s_har_voltarget_trend, s_trend_bb_reentry, simulate, metrics,
    INIT, FEE, MAX_W
)

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# === Load datasets ===
btc = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
btc.index = pd.to_datetime(btc.index)
if btc.index.tz is not None: btc.index = btc.index.tz_localize(None)
btc_returns = btc['r_actual']
btc_prices = (1 + btc_returns).cumprod() * 100
btc_sigma = np.sqrt(np.exp(btc['mu_pred']))

# ETH daily — geen HAR, alleen prijs
eth = pd.read_parquet(P/'data/raw/candles_ETH-EUR_1d.parquet')
eth['date'] = pd.to_datetime(eth['ts'], unit='ms', utc=True).dt.tz_localize(None).dt.normalize()
eth = eth.set_index('date').sort_index()
eth_close = eth['close']
eth_returns = eth_close.pct_change().dropna()
# Align to BTC OOS-periode
eth_returns = eth_returns[(eth_returns.index >= btc_returns.index.min()) & 
                            (eth_returns.index <= btc_returns.index.max())]
eth_prices = (1 + eth_returns).cumprod() * 100

# Sub-period split (zelfde als G5)
split = btc_returns.index[len(btc_returns) // 2]
print(f"BTC Early: {btc_returns[btc_returns.index < split].index.min().date()} → "
      f"{btc_returns[btc_returns.index < split].index.max().date()} "
      f"(n={(btc_returns.index < split).sum()})")
print(f"BTC Late:  {btc_returns[btc_returns.index >= split].index.min().date()} → "
      f"{btc_returns[btc_returns.index >= split].index.max().date()} "
      f"(n={(btc_returns.index >= split).sum()})")
print(f"ETH Full:  {eth_returns.index.min().date()} → {eth_returns.index.max().date()} "
      f"(n={len(eth_returns)})")


def run_strategy(name, fn, returns, prices, sigma_pred, requires_har):
    """Returns Sharpe, AnnRet, MDD for given strategy + data."""
    if requires_har and sigma_pred is None:
        return None
    weights = fn(prices, returns, sigma_pred)
    pv, fees, nr, ns = simulate(weights, returns)
    m = metrics(pv, INIT)
    return m


STRATS = [
    ('B&H',                      s_buyhold,             False),
    ('Trend MA50',               s_trend_ma50,          False),
    ('Bollinger MR',             s_bollinger_mr,        False),
    ('Bollinger Breakout',       s_bollinger_bo,        False),
    ('MACD',                     s_macd,                False),
    ('ATR-stop + Trend MA50',    s_atr_stop,            False),
    ('Donchian',                 s_donchian,            False),
    ('RSI-Bounce',               s_rsi_bounce,          False),
    ('HAR vol-regime + Trend',   s_har_volregime_trend, True),
    ('HAR vol-target + Trend',   s_har_voltarget_trend, True),
    ('Trend MA50 + BB pullback', s_trend_bb_reentry,    False),
]

print("\n" + "="*115)
print("H3 — Robustness scorecard")
print("="*115)
print(f"  {'Strategy':<27}{'BTC Full':>13}{'BTC Early':>13}{'BTC Late':>13}"
      f"{'ETH Full':>13}{'Verdict':>15}")
print(f"  {'':<27}{'(Sh|AR)':>13}{'(Sh|AR)':>13}{'(Sh|AR)':>13}{'(Sh|AR)':>13}")
print("-"*115)

results = []
btc_early_ret = btc_returns[btc_returns.index < split]
btc_late_ret  = btc_returns[btc_returns.index >= split]
btc_early_px  = (1 + btc_early_ret).cumprod() * 100
btc_late_px   = (1 + btc_late_ret).cumprod() * 100
btc_early_sig = btc_sigma[btc_sigma.index < split]
btc_late_sig  = btc_sigma[btc_sigma.index >= split]

def short(m):
    if m is None: return '   N/A      '
    return f"{m['sharpe']:+.2f}|{m['ann_ret']*100:+5.1f}%"

def verdict(m_full, m_early, m_late, m_eth, requires_har):
    """ROBUST = positive Sharpe in alle relevante periodes.
    CONDITIONAL = positive in sommige.
    FRAGILE = negative of zero in meeste periodes."""
    sharpes = [m['sharpe'] if m else None for m in [m_full, m_early, m_late, m_eth]]
    if requires_har:
        sharpes = sharpes[:3]  # geen ETH voor HAR-strats
    pos = sum(1 for s in sharpes if s is not None and s > 0.3)
    nonneg = sum(1 for s in sharpes if s is not None and s > 0)
    total = sum(1 for s in sharpes if s is not None)
    if pos == total: return 'ROBUST'
    if nonneg == total and pos >= total - 1: return 'CONDITIONAL'
    return 'FRAGILE'

for name, fn, req_har in STRATS:
    m_full  = run_strategy(name, fn, btc_returns, btc_prices, btc_sigma, req_har)
    m_early = run_strategy(name, fn, btc_early_ret, btc_early_px, btc_early_sig, req_har)
    m_late  = run_strategy(name, fn, btc_late_ret, btc_late_px, btc_late_sig, req_har)
    if req_har:
        m_eth = None
    else:
        m_eth = run_strategy(name, fn, eth_returns, eth_prices, None, False)
    v = verdict(m_full, m_early, m_late, m_eth, req_har)
    results.append({'strategy': name, 'full': m_full, 'early': m_early,
                    'late': m_late, 'eth': m_eth, 'verdict': v,
                    'requires_har': req_har})
    print(f"  {name:<27}{short(m_full):>13}{short(m_early):>13}{short(m_late):>13}"
          f"{short(m_eth):>13}{v:>15}")

print("-"*115)
print("(ETH = N/A voor HAR-strategieën omdat HAR niet getraind is op ETH-EUR)")

# === ATR-multiplier sensitivity (alleen voor ATR-stop op BTC full) ===
print("\n" + "="*60)
print("ATR-stop: multiplier sensitivity (BTC-EUR full OOS, MA50 entry)")
print("="*60)
print(f"  {'ATR-mult':>10}{'Final':>10}{'AnnRet':>10}{'Sharpe':>10}{'MDD':>10}{'#Reb':>8}")

multipliers = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
atr_results = []
for mult in multipliers:
    def fn(p, r, s, mult=mult):
        return s_atr_stop(p, r, s, ma_window=50, atr_window=14, atr_mult=mult)
    weights = fn(btc_prices, btc_returns, btc_sigma)
    pv, fees, nr, ns = simulate(weights, btc_returns)
    m = metrics(pv, INIT)
    atr_results.append({'mult': mult, **m, 'n_rebal': nr, 'fees': fees})
    print(f"  {mult:>10.1f}{m['final']:>10.2f}{m['ann_ret']*100:>+9.1f}%"
          f"{m['sharpe']:>+10.2f}{m['mdd']*100:>+9.1f}%{nr:>8d}")

# Save results
results_flat = []
for r in results:
    row = {'strategy': r['strategy'], 'requires_har': r['requires_har'],
            'verdict': r['verdict']}
    for period in ['full', 'early', 'late', 'eth']:
        m = r[period]
        if m:
            row[f'{period}_sharpe'] = m['sharpe']
            row[f'{period}_annret'] = m['ann_ret']
            row[f'{period}_mdd'] = m['mdd']
    results_flat.append(row)
pd.DataFrame(results_flat).to_csv(P/'outputs/tables/H3_robustness_scorecard.csv', index=False)
pd.DataFrame(atr_results).to_csv(P/'outputs/tables/H3_atr_multiplier_sweep.csv', index=False)
print(f"\n✓ Resultaten: outputs/tables/H3_robustness_scorecard.csv + H3_atr_multiplier_sweep.csv")

# === Plot scorecard heatmap ===
import numpy as np
strat_names = [r['strategy'] for r in results]
periods = ['BTC Full', 'BTC Early', 'BTC Late', 'ETH Full']
data = np.full((len(strat_names), 4), np.nan)
for i, r in enumerate(results):
    for j, key in enumerate(['full', 'early', 'late', 'eth']):
        if r[key] is not None:
            data[i, j] = r[key]['sharpe']

fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                          gridspec_kw={'width_ratios': [2.2, 1]})

# Heatmap
im = axes[0].imshow(data, cmap='RdYlGn', vmin=-0.5, vmax=1.5, aspect='auto')
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(periods)
axes[0].set_yticks(range(len(strat_names))); axes[0].set_yticklabels(strat_names)
axes[0].set_title('Sharpe Robustness Scorecard', fontweight='bold')
for i in range(len(strat_names)):
    for j in range(4):
        v = data[i,j]
        if np.isnan(v):
            axes[0].text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')
        else:
            color = 'white' if abs(v) > 0.8 else 'black'
            axes[0].text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=10, color=color)
fig.colorbar(im, ax=axes[0], label='Net Sharpe')

# ATR multiplier sweep
ax = axes[1]
mults = [r['mult'] for r in atr_results]
sharpes = [r['sharpe'] for r in atr_results]
rets = [r['ann_ret']*100 for r in atr_results]
ax.plot(mults, sharpes, 'o-', color='#dc2626', lw=2, markersize=8)
ax.axhline(0.90, ls=':', color='gray', label='Trend MA50 baseline (0.90)')
ax.set_xlabel('ATR multiplier'); ax.set_ylabel('Net Sharpe')
ax.set_title('ATR-stop: multiplier sensitivity', fontweight='bold')
ax.grid(True, alpha=0.3); ax.legend()
for m, sh, ret in zip(mults, sharpes, rets):
    ax.annotate(f'{sh:.2f}\n({ret:+.0f}%)', (m, sh), 
                  textcoords='offset points', xytext=(0, 10),
                  fontsize=8, ha='center')

fig.tight_layout()
fig.savefig(P/'outputs/figures/H3_robustness_scorecard.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"✓ Figuur: outputs/figures/H3_robustness_scorecard.png")

# === Final summary ===
print("\n" + "="*70)
print("EINDOORDEEL — robuust over alle relevante periodes:")
print("="*70)
for r in sorted(results, key=lambda x: x['full']['sharpe'] if x['full'] else -1,
                  reverse=True):
    if r['verdict'] == 'ROBUST':
        print(f"  ✓ {r['strategy']:<30} (verdict: {r['verdict']})")
print()
for r in sorted(results, key=lambda x: x['full']['sharpe'] if x['full'] else -1,
                  reverse=True):
    if r['verdict'] == 'CONDITIONAL':
        print(f"  ~ {r['strategy']:<30} (verdict: {r['verdict']})")
print()
for r in sorted(results, key=lambda x: x['full']['sharpe'] if x['full'] else -1,
                  reverse=True):
    if r['verdict'] == 'FRAGILE':
        print(f"  ✗ {r['strategy']:<30} (verdict: {r['verdict']})")
