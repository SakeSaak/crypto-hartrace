"""H4 — Expected Shortfall analyse: de canonical HAR-toepassing.

Sinds Basel III (2016) is ES (Expected Shortfall) de mandated risk-meet,
niet meer VaR alleen. ES_α = E[r | r ≤ VaR_α].

We tonen drie dingen:
  1. ES-forecasts op basis van HAR-σ̂ (Normal en Student-t aannames)
  2. Realized ES op tail-violation dagen
  3. Acerbi-Szekely (2014) Z1-backtest voor ES adequacy

Dit is de juiste HAR-toepassing: variance-forecast → density → ES.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# Load HAR-RS-DOW walk-forward forecasts
df = pd.read_parquet(P / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# HAR voorspelt log-RK; daarvan σ-return-forecast:
# RK_{t+1} = exp(log_RK_{t+1}), σ_{t+1} = sqrt(RK_{t+1})
# Approx: E[σ_{t+1}] ≈ sqrt(exp(mu_pred + 0.5*sigma_pred^2)) (lognormal correction)
df['sigma_ret'] = np.sqrt(np.exp(df['mu_pred'] + 0.5 * df['sigma_pred']**2))

# === Compute ES forecasts ===
def es_normal(sigma, alpha):
    """ES_α voor Normal distribution: ES = σ * φ(z_α) / α (left tail)."""
    z = stats.norm.ppf(alpha)
    return -sigma * stats.norm.pdf(z) / alpha

def es_studentt(sigma, alpha, nu):
    """ES_α voor standardized Student-t (variance=1).
    
    Formula: ES_α = -σ * (ν + t_α²)/(ν-1) * f_ν(t_α)/α / sqrt(ν/(ν-2))
    
    Waarbij t_α = inv-cdf van standard t_ν op niveau α (negative).
    De standardisatie factor sqrt(ν/(ν-2)) zorgt dat var(Z)=1.
    """
    if nu <= 2:
        return np.full_like(sigma, np.nan)
    t_alpha = stats.t.ppf(alpha, nu)  # negative for left tail
    pdf_t = stats.t.pdf(t_alpha, nu)
    # ES voor unstandardized t_ν
    es_t = (nu + t_alpha**2) / (nu - 1) * pdf_t / alpha
    # Scale to standardized t (variance=1) and to current σ
    standardize = np.sqrt(nu / (nu - 2))
    return -sigma * es_t / standardize

# Use ν=3 (left-tail-heavy crypto returns) and ν=5 (moderate) — sensitivity
NU_RET = 3  # conservative tail assumption for crypto

for alpha in [0.01, 0.05]:
    df[f'es_norm_{int(alpha*100):02d}']  = es_normal(df['sigma_ret'], alpha)
    df[f'es_t3_{int(alpha*100):02d}']    = es_studentt(df['sigma_ret'], alpha, NU_RET)
    df[f'es_t5_{int(alpha*100):02d}']    = es_studentt(df['sigma_ret'], alpha, 5)

# === Realized ES: tail violations ===
print("=" * 90)
print("H4 — Expected Shortfall analyse op BTC-EUR HAR-RS-DOW")
print("=" * 90)
print(f"OOS sample: {df.index[0].date()} → {df.index[-1].date()} (n={len(df)})")
print(f"Skewed-t parameters (constant uit MLE op log_RK residuen):")
print(f"  ν = {df['nu_pred'].iloc[0]:.2f}, λ = {df['lam_pred'].iloc[0]:.3f}")
print(f"Assumption voor return-density: r_t = σ̂_t × Z_t, Z_t ~ std-Student-t(ν=3)")
print()

print(f"{'Niveau α':<12}{'Forecast ES (avg)':>25}{'Realized ES':>17}{'Z1-statistiek':>18}{'Verdict':>14}")
print("-" * 90)

results = []
for alpha in [0.01, 0.05]:
    # Tail-events: returns ≤ VaR_α
    var_col = f'var_return_{int(alpha*100):02d}'
    viol_col = f'viol_return_{int(alpha*100):02d}'
    tail_returns = df.loc[df[viol_col] == 1, 'r_actual']
    realized_es = tail_returns.mean() if len(tail_returns) > 0 else np.nan
    n_viol = len(tail_returns)
    n_exp = int(alpha * len(df))
    
    # Acerbi-Szekely Z1 test: 
    # Z1 = (1/N_α) Σ (r_t / ES_α(t)) over violating days, minus 1
    # Onder H0 (model correct): E[Z1] = 0
    # Reject if Z1 << 0 (realized losses worse than predicted)
    forecast_es_t3 = df.loc[df[viol_col] == 1, f'es_t3_{int(alpha*100):02d}']
    if len(tail_returns) > 0:
        z1 = (tail_returns / forecast_es_t3).mean() + 1
    else:
        z1 = np.nan
    
    # Avg forecasted ES over hele sample (voor vergelijking met realized)
    avg_es_t3 = df[f'es_t3_{int(alpha*100):02d}'].mean()
    
    verdict = ''
    if not np.isnan(z1):
        # Negative Z1 = realized losses GROTER dan ES voorspelt = model underestimates risk
        if z1 < -0.10:
            verdict = '⚠ underforecast'
        elif z1 > 0.10:
            verdict = '✓ conservatief'
        else:
            verdict = '✓ calibrated'
    
    print(f"  ES_{int(alpha*100):02d}%      {avg_es_t3*100:>23.2f}%{realized_es*100:>15.2f}%"
          f"{z1:>17.3f}{verdict:>16}")
    results.append({'alpha': alpha, 'n_viol': n_viol, 'n_exp': n_exp,
                     'forecast_es_t3': avg_es_t3, 'realized_es': realized_es,
                     'z1': z1, 'verdict': verdict})
    print(f"     n_violations: {n_viol} (verwacht {n_exp}, "
          f"realised/expected ratio = {n_viol/n_exp:.2f})")

# Save full ES forecast series for thesis
df_out = df[['r_actual', 'sigma_ret',
              'es_norm_01', 'es_t3_01', 'es_t5_01',
              'es_norm_05', 'es_t3_05', 'es_t5_05',
              'var_return_01', 'viol_return_01',
              'var_return_05', 'viol_return_05']].copy()
df_out.to_csv(P / 'outputs/tables/H4_es_forecasts.csv')
pd.DataFrame(results).to_csv(P / 'outputs/tables/H4_es_summary.csv', index=False)

# === Plot ES forecasts vs realized in tail-days ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for i, alpha in enumerate([0.01, 0.05]):
    ax = axes[i]
    viol_col = f'viol_return_{int(alpha*100):02d}'
    es_col = f'es_t3_{int(alpha*100):02d}'
    
    # Time-series
    ax.plot(df.index, df['r_actual']*100, color='gray', alpha=0.4, lw=0.5, label='Realized return')
    ax.plot(df.index, df[es_col]*100, color='#dc2626', lw=1.2, 
            label=f'ES_{int(alpha*100):02d}% forecast (t-distr ν=3)')
    
    # Violation days
    viol_days = df[df[viol_col] == 1]
    ax.scatter(viol_days.index, viol_days['r_actual']*100, 
               s=30, color='#7c3aed', edgecolor='black', zorder=5,
               label=f'Tail event (n={len(viol_days)})')
    
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.set_ylabel(f'Return (%)')
    ax.set_title(f'Expected Shortfall α={int(alpha*100)}% — Z1={results[i]["z1"]:.3f}',
                  fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('H4 — Expected Shortfall: HAR-driven density risk-forecast op BTC-EUR',
              fontweight='bold', fontsize=13)
fig.tight_layout()
fig.savefig(P / 'outputs/figures/H4_expected_shortfall.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Resultaten: outputs/tables/H4_es_forecasts.csv (n={len(df_out)} dagen)")
print(f"✓ Summary:    outputs/tables/H4_es_summary.csv")
print(f"✓ Figuur:     outputs/figures/H4_expected_shortfall.png")
print()
print("INTERPRETATIE:")
for r in results:
    if r['z1'] < -0.10:
        what = "model onderschat staartrisico → meer kapitaalbuffer nodig in echte Basel-context"
    elif r['z1'] > 0.10:
        what = "model is conservatief — overschat staartverlies enigszins"
    else:
        what = "model is goed gekalibreerd — realized en forecasted ES komen overeen"
    print(f"  ES_{int(r['alpha']*100):02d}%: {what}")
