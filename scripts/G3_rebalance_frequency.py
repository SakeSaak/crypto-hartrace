"""G3 — Rebalance-frequency optimization op €110-schaal.

Vraag: bij welke rebalance-frequentie (1, 2, 3, 5, 7, 14, 22 dagen) krijgen
we de beste netto Sharpe op een klein account (€110-1000), waar TC en
min-trade-€5 een grote rol spelen?

Aanpak: gebruik HAR-RS-Q-WE-X daily forecasts (E1 walk-forward), simuleer
strategieën die alleen elke N-de dag rebalancen. Op niet-rebalance-dagen
hou je positie vast. Inclusief €5-min-trade skip-logica.
"""
from __future__ import annotations
import sys; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')
FEE_MAKER = 0.0015
MIN_TRADE_EUR = 5.0
MAX_ALLOC = 0.90
TARGET_VOL = 0.35


def vol_managed_freq(sigma_pred, returns, target_vol_ann, fee,
                       max_alloc, rebal_every_n_days, initial_eur,
                       min_trade_eur):
    """Vol-managed met N-day rebalance + €5 min trade skip-logica.
    
    Simuleert €-niveau om min-trade effect te capturen.
    Returns: DataFrame met portfolio_eur, w_realized, turnover per dag.
    """
    target_d = target_vol_ann / np.sqrt(365)
    target_w = (target_d / sigma_pred).clip(upper=max_alloc, lower=0.0)
    
    n = len(sigma_pred)
    portfolio = initial_eur
    btc_value = 0.0
    eur_value = initial_eur
    
    history = {'portfolio': [], 'w': [], 'turnover_eur': [], 
               'fees': [], 'skipped': []}
    
    for i in range(n):
        # Apply day i's return on BTC part
        btc_value = btc_value * (1 + returns.iloc[i])
        portfolio = btc_value + eur_value
        current_w = btc_value / portfolio if portfolio > 0 else 0
        
        # Rebalance check
        is_rebal_day = (i % rebal_every_n_days == 0)
        fee_paid = 0.0; skipped = False
        
        if is_rebal_day:
            target_eur = float(target_w.iloc[i]) * portfolio
            delta_eur = target_eur - btc_value
            
            if abs(delta_eur) < min_trade_eur:
                # Skip — sub-minimum trade
                skipped = True
            else:
                # Execute
                fee_paid = abs(delta_eur) * fee
                btc_value = target_eur
                eur_value = portfolio - btc_value - fee_paid
                portfolio = btc_value + eur_value
        
        history['portfolio'].append(portfolio)
        history['w'].append(btc_value / portfolio if portfolio > 0 else 0)
        history['turnover_eur'].append(abs(delta_eur) if (is_rebal_day and not skipped) else 0)
        history['fees'].append(fee_paid)
        history['skipped'].append(skipped)
    
    df = pd.DataFrame(history, index=sigma_pred.index)
    return df


def metrics(portfolio_series, initial):
    r = portfolio_series.pct_change().dropna()
    n = len(r)
    final = float(portfolio_series.iloc[-1])
    tot_ret = (final / initial) - 1
    ann_ret = (final / initial) ** (365/n) - 1 if n > 0 else 0
    ann_vol = float(r.std(ddof=1)) * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dn = r[r<0]
    sortino = ann_ret / (dn.std(ddof=1) * np.sqrt(365)) if len(dn)>1 else 0
    eq = portfolio_series
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return dict(final=final, tot_ret=tot_ret, ann_ret=ann_ret,
                ann_vol=ann_vol, sharpe=sharpe, sortino=sortino, mdd=mdd,
                calmar=ann_ret/abs(mdd) if mdd<0 else 0)


# Load HAR-RS-Q-WE-X forecasts (G2 winner)
df = pd.read_parquet(f'{PROJECT}/outputs/tables/E1_walkforward_per_day_HAR-RS-Q-WE-X.parquet')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)
sig = np.sqrt(np.exp(df['mu_pred']))
ret = df['r_actual']

print("="*100)
print("G3 — Rebalance-frequentie sweep voor HAR-RS-Q-WE-X (vol-managed, long-only)")
print("="*100)
print(f"Periode: {df.index.min().date()} → {df.index.max().date()} (n={len(df)} dagen)")
print(f"Config: target_vol={TARGET_VOL:.0%}, max_alloc={MAX_ALLOC:.0%}, "
      f"fee=15bps maker, min_trade=€{MIN_TRADE_EUR}")
print()

# Test verschillende rebalance-frequenties op twee account-sizes
for initial in [110, 1000]:
    print(f"\n{'─'*100}")
    print(f"  Startkapitaal: €{initial}")
    print(f"{'─'*100}")
    print(f"{'Freq':>10}{'Final':>10}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}"
          f"{'Sharpe':>8}{'MDD':>9}{'#Rebal':>8}{'#Skip':>7}{'TotFees':>9}{'Fee%':>7}")
    print(f"{'─'*100}")
    
    # Baseline: buy-and-hold
    bh_pv = initial * (1 + ret).cumprod()
    m = metrics(bh_pv, initial)
    print(f"{'B&H':>10}{m['final']:>10.2f}{m['tot_ret']*100:>+8.1f}%"
          f"{m['ann_ret']*100:>+8.1f}%{m['ann_vol']*100:>8.1f}%"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>+8.1f}%{'0':>8}{'0':>7}{'0.00':>9}{'0.0%':>7}")
    
    rows = []
    for N in [1, 2, 3, 5, 7, 14, 22]:
        h = vol_managed_freq(sig, ret, TARGET_VOL, FEE_MAKER, MAX_ALLOC,
                              N, initial, MIN_TRADE_EUR)
        n_rebal = (h['turnover_eur'] > 0).sum()
        n_skip = h['skipped'].sum()
        tot_fees = h['fees'].sum()
        m = metrics(h['portfolio'], initial)
        fee_pct = tot_fees / initial * 100
        rows.append({'N': N, **m, 'n_rebal': n_rebal, 'n_skip': n_skip,
                     'tot_fees': tot_fees, 'fee_pct': fee_pct, 'history': h})
        print(f"{N:>9}d{m['final']:>10.2f}{m['tot_ret']*100:>+8.1f}%"
              f"{m['ann_ret']*100:>+8.1f}%{m['ann_vol']*100:>8.1f}%"
              f"{m['sharpe']:>8.2f}{m['mdd']*100:>+8.1f}%{n_rebal:>8d}{n_skip:>7d}"
              f"{tot_fees:>9.2f}{fee_pct:>6.1f}%")
    
    # Best by Sharpe
    best = max(rows, key=lambda r: r['sharpe'])
    print(f"\n  Best Sharpe: rebalance elke {best['N']} dagen "
          f"(Sharpe={best['sharpe']:.3f}, AnnRet={best['ann_ret']*100:+.1f}%, "
          f"fees=€{best['tot_fees']:.2f} = {best['fee_pct']:.1f}%)")
    
    # Save plot for €110 case
    if initial == 110:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(bh_pv.index, bh_pv.values, lw=1.2, color='#6b7280',
                 label=f'Buy-and-hold (Sharpe={metrics(bh_pv, initial)["sharpe"]:.2f})')
        for r in rows:
            if r['N'] in [1, 3, 7, 14]:
                ax.plot(r['history'].index, r['history']['portfolio'].values,
                         lw=1.4, label=f"N={r['N']}d (Sh={r['sharpe']:.2f}, "
                                       f"fees €{r['tot_fees']:.2f})")
        ax.set_yscale('log')
        ax.set_title('Rebalance-frequentie effect op €110 portfolio (HAR-RS-Q-WE-X)',
                      fontweight='bold')
        ax.set_xlabel('Datum'); ax.set_ylabel('Portfolio waarde (€)')
        ax.grid(True, alpha=0.3, which='both'); ax.legend(loc='upper left')
        fig.tight_layout()
        fig.savefig(f'{PROJECT}/outputs/figures/G3_rebalance_freq_110eur.png',
                     dpi=130, bbox_inches='tight')
        plt.close(fig)

print()
print(f"Saved: outputs/figures/G3_rebalance_freq_110eur.png")
