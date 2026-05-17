"""G5 — Robustness checks voor Trend MA50-bevinding uit G4.

Drie onafhankelijke checks:
  1. Sub-period stability: 2022-10 → 2024-05 (early) vs 2024-05 → 2026-05 (late)
  2. MA-window sweep: 20, 30, 40, ..., 200 dagen
  3. Cross-asset transfer: ETH-EUR met dezelfde MA50-regel
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
MAX_W = 0.90


def pure_trend(returns, ma_window, init=110, max_w=MAX_W):
    """Pure trend filter: full position if price > MA, cash otherwise."""
    price = (1 + returns).cumprod()
    ma = price.rolling(ma_window, min_periods=10).mean()
    in_trend = (price > ma).fillna(True).values
    
    pv, btc, eur = init, 0.0, init
    pvs, fees, n_rebal = [], 0.0, 0
    for i in range(len(returns)):
        btc = btc * (1 + returns.iloc[i]); pv = btc + eur
        current_w = btc / pv if pv > 0 else 0
        target_w = max_w if in_trend[i] else 0.0
        if abs(target_w - current_w) > 0.05:  # 5% deadband
            target_eur = target_w * pv
            d = target_eur - btc
            if abs(d) >= MIN_TRADE:
                fee_p = abs(d) * FEE
                btc, eur = target_eur, pv - target_eur - fee_p
                pv = btc + eur
                fees += fee_p; n_rebal += 1
        pvs.append(pv)
    return pd.Series(pvs, index=returns.index), fees, n_rebal


def buy_hold(returns, init=110, w=MAX_W):
    return init * w * (1 + returns).cumprod() + init * (1 - w)


def stats(pv, init):
    r = pv.pct_change().dropna(); n = len(r)
    if n == 0 or pv.iloc[-1] <= 0: return None
    final = float(pv.iloc[-1])
    ar = (final/init)**(365/n) - 1
    av = float(r.std(ddof=1)) * np.sqrt(365)
    sh = ar/av if av > 0 else 0
    mdd = float(((pv-pv.cummax())/pv.cummax()).min())
    return dict(final=final, ann_ret=ar, ann_vol=av, sharpe=sh, mdd=mdd,
                calmar=ar/abs(mdd) if mdd<0 else 0)


# Load BTC daily returns (from E1 walk-forward, OOS sample 2022-10 to 2026-05)
df_btc = pd.read_parquet(f'{P}/outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df_btc.index = pd.to_datetime(df_btc.index)
if df_btc.index.tz is not None: df_btc.index = df_btc.index.tz_localize(None)
ret_btc = df_btc['r_actual']

# ===== CHECK 1: Sub-period stability =====
print("=" * 100)
print("CHECK 1 — Sub-period stability (BTC-EUR)")
print("=" * 100)

split = ret_btc.index[len(ret_btc) // 2]
ret_early = ret_btc[ret_btc.index < split]
ret_late = ret_btc[ret_btc.index >= split]
print(f"Early:  {ret_early.index.min().date()} → {ret_early.index.max().date()} (n={len(ret_early)})")
print(f"Late:   {ret_late.index.min().date()} → {ret_late.index.max().date()} (n={len(ret_late)})")
print()

for sub_label, sub_ret in [('Early (2022-2024)', ret_early), ('Late (2024-2026)', ret_late)]:
    print(f"  {sub_label}:")
    pv_bh = buy_hold(sub_ret, init=110)
    s_bh = stats(pv_bh, 110)
    print(f"    {'B&H':<25} Final=€{s_bh['final']:6.2f}, AnnRet={s_bh['ann_ret']*100:+6.1f}%, "
          f"Sharpe={s_bh['sharpe']:+.2f}, MDD={s_bh['mdd']*100:+.1f}%")
    
    pv_t, fees, nr = pure_trend(sub_ret, 50, init=110)
    s_t = stats(pv_t, 110)
    print(f"    {'Trend MA50':<25} Final=€{s_t['final']:6.2f}, AnnRet={s_t['ann_ret']*100:+6.1f}%, "
          f"Sharpe={s_t['sharpe']:+.2f}, MDD={s_t['mdd']*100:+.1f}%, Fees=€{fees:.2f}, #Reb={nr}")
    
    pv_t2, fees2, nr2 = pure_trend(sub_ret, 200, init=110)
    s_t2 = stats(pv_t2, 110)
    print(f"    {'Trend MA200':<25} Final=€{s_t2['final']:6.2f}, AnnRet={s_t2['ann_ret']*100:+6.1f}%, "
          f"Sharpe={s_t2['sharpe']:+.2f}, MDD={s_t2['mdd']*100:+.1f}%, Fees=€{fees2:.2f}, #Reb={nr2}")
    print()


# ===== CHECK 2: MA-window sweep =====
print("=" * 100)
print("CHECK 2 — MA-window sweep (BTC-EUR full OOS-periode)")
print("=" * 100)
print(f"{'MA-window':>10}{'Final':>9}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>8}{'MDD':>9}{'#Rebal':>8}{'Fees':>8}")
print("-" * 80)

sweep_results = []
for w in [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 180, 200, 250]:
    pv, fees, nr = pure_trend(ret_btc, w, init=110)
    s = stats(pv, 110)
    if s:
        sweep_results.append({'window': w, **s, 'fees': fees, 'n_rebal': nr})
        print(f"{w:>10}{s['final']:>9.2f}{s['ann_ret']*100:>+8.1f}%{s['ann_vol']*100:>8.1f}%"
              f"{s['sharpe']:>+8.2f}{s['mdd']*100:>+8.1f}%{nr:>8d}{fees:>8.2f}")

# Plot the sweep
sweep_df = pd.DataFrame(sweep_results)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(sweep_df['window'], sweep_df['sharpe'], 'o-', color='#dc2626', lw=2)
axes[0].axhline(0.50, ls=':', color='gray', label='B&H Sharpe = 0.50')
axes[0].set_xlabel('MA window (days)'); axes[0].set_ylabel('Net Sharpe')
axes[0].set_title('Sharpe vs MA-window'); axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot(sweep_df['window'], sweep_df['ann_ret']*100, 'o-', color='#059669', lw=2)
axes[1].axhline(21.9, ls=':', color='gray', label='B&H AnnRet')
axes[1].set_xlabel('MA window (days)'); axes[1].set_ylabel('Annualized Return (%)')
axes[1].set_title('Return vs MA-window'); axes[1].grid(True, alpha=0.3); axes[1].legend()

axes[2].plot(sweep_df['window'], sweep_df['mdd']*100, 'o-', color='#7c3aed', lw=2)
axes[2].axhline(-56.5, ls=':', color='gray', label='B&H MDD')
axes[2].set_xlabel('MA window (days)'); axes[2].set_ylabel('Max Drawdown (%)')
axes[2].set_title('MDD vs MA-window'); axes[2].grid(True, alpha=0.3); axes[2].legend()

fig.suptitle('CHECK 2: Robustness van MA-window keuze (pure trend, BTC-EUR)',
              fontweight='bold')
fig.tight_layout()
fig.savefig(f'{P}/outputs/figures/G5_ma_window_sweep.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ===== CHECK 3: ETH-EUR cross-asset =====
print("\n" + "=" * 100)
print("CHECK 3 — Cross-asset transfer: ETH-EUR (zelfde MA50-regel, geen retraining)")
print("=" * 100)

eth_daily = pd.read_parquet(f'{P}/data/raw/candles_ETH-EUR_1d.parquet')
eth_daily['date'] = pd.to_datetime(eth_daily['ts'], unit='ms', utc=True).dt.tz_localize(None).dt.normalize()
eth_daily = eth_daily.set_index('date').sort_index()
eth_ret = eth_daily['close'].pct_change().dropna()

# Align to same OOS-periode als BTC voor eerlijke vergelijking
eth_ret_aligned = eth_ret[(eth_ret.index >= ret_btc.index.min()) & (eth_ret.index <= ret_btc.index.max())]
print(f"ETH-EUR aligned: {len(eth_ret_aligned)} dagen "
      f"({eth_ret_aligned.index.min().date()} → {eth_ret_aligned.index.max().date()})")
print()

# B&H ETH
pv_eth_bh = buy_hold(eth_ret_aligned, init=110)
s = stats(pv_eth_bh, 110)
print(f"  {'ETH B&H':<25} Final=€{s['final']:6.2f}, AnnRet={s['ann_ret']*100:+6.1f}%, "
      f"Sharpe={s['sharpe']:+.2f}, MDD={s['mdd']*100:+.1f}%")

# Trend MA50/100/200 on ETH
for w in [50, 100, 200]:
    pv_eth_t, fees, nr = pure_trend(eth_ret_aligned, w, init=110)
    s = stats(pv_eth_t, 110)
    print(f"  {'ETH Trend MA'+str(w):<25} Final=€{s['final']:6.2f}, AnnRet={s['ann_ret']*100:+6.1f}%, "
          f"Sharpe={s['sharpe']:+.2f}, MDD={s['mdd']*100:+.1f}%, Fees=€{fees:.2f}, #Reb={nr}")

# ===== Combined view: equity curves =====
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Top-left: BTC sub-periods
ax = axes[0, 0]
ax.plot(buy_hold(ret_early, 110).index, buy_hold(ret_early, 110).values,
         lw=1.2, color='gray', label='Early B&H', alpha=0.6)
ax.plot(pure_trend(ret_early, 50)[0].index, pure_trend(ret_early, 50)[0].values,
         lw=1.6, color='#dc2626', label='Early Trend MA50')
ax.set_title('BTC: Early period (2022-10 → 2024-05)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_yscale('log'); ax.set_ylabel('€')

ax = axes[0, 1]
ax.plot(buy_hold(ret_late, 110).index, buy_hold(ret_late, 110).values,
         lw=1.2, color='gray', label='Late B&H', alpha=0.6)
ax.plot(pure_trend(ret_late, 50)[0].index, pure_trend(ret_late, 50)[0].values,
         lw=1.6, color='#dc2626', label='Late Trend MA50')
ax.set_title('BTC: Late period (2024-05 → 2026-05)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_yscale('log'); ax.set_ylabel('€')

# Bottom-left: ETH full
ax = axes[1, 0]
ax.plot(buy_hold(eth_ret_aligned, 110).index, buy_hold(eth_ret_aligned, 110).values,
         lw=1.2, color='gray', label='ETH B&H', alpha=0.6)
ax.plot(pure_trend(eth_ret_aligned, 50)[0].index, pure_trend(eth_ret_aligned, 50)[0].values,
         lw=1.6, color='#dc2626', label='ETH Trend MA50')
ax.set_title('ETH-EUR: full OOS period (cross-asset)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_yscale('log'); ax.set_ylabel('€')

# Bottom-right: MA window plateau zoom
ax = axes[1, 1]
ax.plot(sweep_df['window'], sweep_df['sharpe'], 'o-', color='#dc2626', lw=2)
ax.axhline(0.50, ls=':', color='gray', label='B&H Sharpe')
ax.fill_between(sweep_df['window'], 0.50, sweep_df['sharpe'],
                  where=sweep_df['sharpe']>0.50, alpha=0.2, color='green', label='Beats B&H')
ax.set_xlabel('MA window (days)'); ax.set_ylabel('Net Sharpe (BTC-EUR)')
ax.set_title('MA-window robustness plateau'); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle('G5 — Robustness checks: sub-period stability, MA-window sweep, ETH-transfer',
              fontweight='bold', fontsize=13)
fig.tight_layout()
fig.savefig(f'{P}/outputs/figures/G5_robustness_overview.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved: outputs/figures/G5_ma_window_sweep.png")
print(f"       outputs/figures/G5_robustness_overview.png")
