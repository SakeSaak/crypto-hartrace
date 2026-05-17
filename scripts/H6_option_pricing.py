"""H6 — HAR-σ als input voor option-pricing.

DE canonical HAR-toepassing: vol-forecasting voor derivatives.

Simulatie: elke dag in OOS, "verkoop" een 7-day ATM straddle (call + put).
Premium = Black-Scholes met σ_model als input.
Realized payoff na 7 dagen = |S_{t+7} - S_t|.
Edge = Premium - Payoff (positief = seller wint geld).

Drie vol-modellen vergeleken:
  1. Constant σ (full-sample std)        ← naïef benchmark
  2. Rolling 30-day realized std         ← simpel-historisch
  3. HAR-RS-DOW voorspelde σ_{t+1}      ← onze model

Hypothese: HAR-σ levert beste straddle-pricing want het is forward-looking
en time-varying. Constant-σ mist het signal, rolling lag het signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# Load data
df = pd.read_parquet(P / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)
df['sigma_ret_har'] = np.sqrt(np.exp(df['mu_pred'] + 0.5 * df['sigma_pred']**2))

# Reconstruct prices (genormaliseerd naar S_0=100)
df['S'] = 100 * (1 + df['r_actual']).cumprod()

# === Black-Scholes formulas ===
def bs_call(S, K, sigma, T, r=0.0):
    """Black-Scholes price for European call. sigma is ANNUALIZED vol."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

def bs_put(S, K, sigma, T, r=0.0):
    """Black-Scholes put via put-call parity."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    return bs_call(S, K, sigma, T, r) - S + K * np.exp(-r*T)

def bs_straddle(S, sigma, T_days):
    """ATM straddle (K=S): call+put. sigma=daily vol, T=days."""
    T_years = T_days / 365
    sigma_ann = sigma * np.sqrt(365)
    return bs_call(S, S, sigma_ann, T_years) + bs_put(S, S, sigma_ann, T_years)

# === Build three σ-series ===
HOLD_DAYS = 7  # 7-day straddle expiry

# 1. Constant σ (full-sample)
sigma_const = df['r_actual'].std()
df['sigma_const'] = sigma_const

# 2. Rolling 30-day realized std (using returns up to t-1)
df['sigma_roll30'] = df['r_actual'].rolling(30, min_periods=10).std().shift(1)
# Fill NaN with constant σ for early period
df['sigma_roll30'] = df['sigma_roll30'].fillna(sigma_const)

# 3. HAR-RS-DOW forecast (already in df)
# σ_ret_har is for daily returns

# === Simulate selling straddles ===
print("="*90)
print("H6 — Option-pricing: HAR-σ als input voor 7-day ATM straddle valuation")
print("="*90)
print(f"OOS period: {df.index[0].date()} → {df.index[-1].date()} (n={len(df)})")
print(f"Strategy: verkoop ATM straddle, hold {HOLD_DAYS} dagen, payoff = |S_T - K|")
print()

results = []
df_sim = df.iloc[:-HOLD_DAYS].copy()  # need future S_t+HOLD for payoff

for sigma_name, sigma_col in [
    ('Constant σ',     'sigma_const'),
    ('Rolling 30d σ',  'sigma_roll30'),
    ('HAR-RS-DOW σ',   'sigma_ret_har'),
]:
    # Premium received elke dag voor verkocht straddle
    df_sim[f'prem_{sigma_name}'] = df_sim.apply(
        lambda r: bs_straddle(r['S'], r[sigma_col], HOLD_DAYS), axis=1)
    # Realized payoff after HOLD_DAYS
    df_sim[f'payoff_{sigma_name}'] = (df.loc[df_sim.index, 'S'] - 
                                        df['S'].shift(-HOLD_DAYS).loc[df_sim.index]).abs()
    # Edge per straddle (positief = seller wint)
    df_sim[f'edge_{sigma_name}'] = df_sim[f'prem_{sigma_name}'] - df_sim[f'payoff_{sigma_name}']
    
    # Stats
    edge = df_sim[f'edge_{sigma_name}']
    avg_prem = df_sim[f'prem_{sigma_name}'].mean()
    avg_payoff = df_sim[f'payoff_{sigma_name}'].mean()
    avg_edge = edge.mean()
    mape = (df_sim[f'prem_{sigma_name}'] - df_sim[f'payoff_{sigma_name}']).abs().mean()
    # Sharpe-achtig: average edge / std edge (per straddle, niet annualized)
    sh = avg_edge / edge.std() if edge.std() > 0 else 0
    # Hit rate: % dagen dat premium > payoff
    hit = (edge > 0).mean()
    
    results.append({
        'sigma_model': sigma_name,
        'avg_premium': avg_prem, 'avg_payoff': avg_payoff,
        'avg_edge': avg_edge, 'mape': mape,
        'sharpe_per_straddle': sh, 'hit_rate': hit
    })

# Print results
print(f"{'σ-model':<18}{'Premium':>10}{'Payoff':>10}{'Edge':>10}{'MAPE':>10}{'Sharpe':>10}{'Hit%':>8}")
print("-"*76)
for r in results:
    print(f"{r['sigma_model']:<18}"
          f"{r['avg_premium']:>10.3f}{r['avg_payoff']:>10.3f}"
          f"{r['avg_edge']:>+10.3f}{r['mape']:>10.3f}"
          f"{r['sharpe_per_straddle']:>+10.3f}{r['hit_rate']*100:>7.1f}%")

print()
print("INTERPRETATIE:")
print("  - MAPE (Mean Absolute Pricing Error): lager = beter de option-prijs")
print("  - Edge ≈ 0 onder efficient market hypothesis")
print("  - Hit% > 50% met positive edge = systematische over-pricing")
print("  - HAR-σ moet beste MAPE hebben als variance-forecast accuraat is")

# Plot: cumulative edge over time per σ-model
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1]})

for r in results:
    edge_col = f'edge_{r["sigma_model"]}'
    cum_edge = df_sim[edge_col].cumsum()
    axes[0].plot(cum_edge.index, cum_edge.values, lw=1.5,
                  label=f'{r["sigma_model"]} (MAPE={r["mape"]:.2f}, Sharpe={r["sharpe_per_straddle"]:+.2f})')

axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_ylabel('Cumulative edge (premium − payoff) per €100 notional')
axes[0].set_title('H6 — Cumulatieve straddle-seller edge over 7-day ATM straddles',
                   fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Bottom: HAR σ-forecast vs realized 7-day move (in %)
axes[1].plot(df.index, df['sigma_ret_har']*100, lw=1, color='#dc2626',
              label='HAR σ̂_{t+1} (daily, %)')
real_mover = (df['S'].shift(-HOLD_DAYS) - df['S']).abs() / df['S'] / np.sqrt(HOLD_DAYS) * 100
axes[1].plot(df.index, real_mover, lw=0.8, color='gray', alpha=0.5,
              label=f'Realized |move| over {HOLD_DAYS}d (annualized %)')
axes[1].set_ylabel('Vol (%/day)')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(P/'outputs/figures/H6_option_pricing.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# Save
pd.DataFrame(results).to_csv(P/'outputs/tables/H6_option_pricing.csv', index=False)
print(f"\n✓ Saved: outputs/tables/H6_option_pricing.csv")
print(f"✓ Figuur: outputs/figures/H6_option_pricing.png")
