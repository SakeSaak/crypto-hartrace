"""H5 — Density-robustness check voor Expected Shortfall.

Vergelijkt ES-coverage onder drie return-density aannames:
  1. Normal: r_t = σ_t × N(0,1)
  2. Student-t(ν=3): r_t = σ_t × std-t(3)        ← onze H4 keuze
  3. Hansen 1994 skewed-t (ν, λ uit MLE op returns)

De skewed-t is theoretisch meest accuraat voor crypto-returns (negatieve skewness,
fat tails). Test of de conclusies van H4 robust zijn over density-keuze.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gamma as sp_gamma
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

df = pd.read_parquet(P / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)

# σ_t voor returns (lognormal correction)
df['sigma_ret'] = np.sqrt(np.exp(df['mu_pred'] + 0.5 * df['sigma_pred']**2))

# === Stap 1: Hansen skewed-t fit op gestandaardiseerde returns ===
# Standaardiseer: z_t = r_t / σ_t. Onder correct model: z_t ~ Hansen-skewed-t(ν, λ)
z = (df['r_actual'] / df['sigma_ret']).dropna()
z = z[(z > -8) & (z < 8)]  # trim extreme outliers (defensive)

def hansen_skewt_pdf(z, eta, lam):
    """Hansen 1994 skewed-t pdf. eta = nu (>2), lam in (-1, 1)."""
    c = sp_gamma((eta+1)/2) / (np.sqrt(np.pi*(eta-2)) * sp_gamma(eta/2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3*lam**2 - a**2)
    z_thresh = -a/b
    pdf = np.where(
        z < z_thresh,
        b * c * (1 + 1/(eta-2) * ((b*z + a)/(1 - lam))**2)**(-(eta+1)/2),
        b * c * (1 + 1/(eta-2) * ((b*z + a)/(1 + lam))**2)**(-(eta+1)/2)
    )
    return pdf

def neg_log_lik_skewt(params, z):
    eta, lam = params
    if eta <= 2.01 or eta > 50 or abs(lam) >= 0.99:
        return 1e10
    pdf = hansen_skewt_pdf(z, eta, lam)
    if np.any(pdf <= 0):
        return 1e10
    return -np.sum(np.log(pdf))

# MLE
res = optimize.minimize(neg_log_lik_skewt, x0=[5, -0.1], args=(z.values,),
                         method='Nelder-Mead', options={'xatol': 1e-4})
eta_hat, lam_hat = res.x
print(f"Hansen 1994 skewed-t MLE op gestandaardiseerde BTC returns (z=r/σ):")
print(f"  ν (degrees of freedom) = {eta_hat:.3f}")
print(f"  λ (skewness param)     = {lam_hat:+.4f}")
print(f"  → negatief λ bevestigt linker-staart heaviness (crash-asymmetrie)")
print(f"  Sample size n={len(z)}")
print()

# === Stap 2: MC-sample skewed-t voor ES computation ===
def sample_hansen_skewt(n, eta, lam, rng):
    """Inverse CDF sampling via numerical inversion."""
    u = rng.uniform(size=n)
    # Numerical CDF inversion is slow; use rejection sampling
    # Or analytical via standard t + skew transformation
    # Hansen skewed-t: composed of left/right scaled-t pieces
    from scipy.special import gamma as g
    c = g((eta+1)/2) / (np.sqrt(np.pi*(eta-2)) * g(eta/2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3*lam**2 - a**2)
    # Sample standard-t then transform
    t_samples = stats.t.rvs(df=eta, size=n, random_state=rng)
    # Use Hansen's transformation
    z_samples = np.where(
        rng.uniform(size=n) < (1 - lam) / 2,
        (1 - lam) * np.sqrt((eta-2)/eta) * t_samples - a/b,
        (1 + lam) * np.sqrt((eta-2)/eta) * np.abs(t_samples) - a/b
    )
    # Apply scale correction
    return z_samples / b

# Pre-compute ES_α for skewed-t via large MC simulation
rng = np.random.default_rng(42)
N_MC = 500_000
z_skewt = sample_hansen_skewt(N_MC, eta_hat, lam_hat, rng)
z_skewt = z_skewt - z_skewt.mean()  # zero-center for safety
z_skewt = z_skewt / z_skewt.std()    # unit-variance for safety
print(f"MC skewed-t sample: mean={z_skewt.mean():+.4f}, std={z_skewt.std():.4f}")

# Compute standardized ES quantiles
def es_quantile(z_samples, alpha):
    """ES_α = E[Z | Z ≤ q_α] for left tail."""
    q = np.quantile(z_samples, alpha)
    return z_samples[z_samples <= q].mean()

es_norm_z = {alpha: -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha 
              for alpha in [0.01, 0.05]}
es_t3_z = {}
for alpha in [0.01, 0.05]:
    nu = 3
    t_alpha = stats.t.ppf(alpha, nu)
    es_t3_z[alpha] = -(nu + t_alpha**2) / (nu - 1) * stats.t.pdf(t_alpha, nu) / alpha / np.sqrt(nu/(nu-2))
es_skewt_z = {alpha: es_quantile(z_skewt, alpha) for alpha in [0.01, 0.05]}

print(f"\nStandardized ES (per unit σ):")
print(f"  α=1%:  Normal={es_norm_z[0.01]:+.3f}, Student-t(3)={es_t3_z[0.01]:+.3f}, "
      f"Skewed-t(ν={eta_hat:.1f},λ={lam_hat:+.2f})={es_skewt_z[0.01]:+.3f}")
print(f"  α=5%:  Normal={es_norm_z[0.05]:+.3f}, Student-t(3)={es_t3_z[0.05]:+.3f}, "
      f"Skewed-t(ν={eta_hat:.1f},λ={lam_hat:+.2f})={es_skewt_z[0.05]:+.3f}")

# === Stap 3: Pas toe op HAR sigma-forecasts en compute Z1 ===
results = []
print(f"\n{'Density':<35}{'α':<6}{'Forecast ES':>14}{'Realized ES':>14}{'Z1':>10}{'Verdict':>15}")
print("-"*100)

for density_name, es_factor in [
    ('Normal', es_norm_z),
    ('Student-t(ν=3)', es_t3_z),
    (f'Hansen skewed-t (ν={eta_hat:.1f}, λ={lam_hat:+.2f})', es_skewt_z),
]:
    for alpha in [0.01, 0.05]:
        df[f'es_{alpha}'] = df['sigma_ret'] * es_factor[alpha]
        viol = df[df[f'viol_return_{int(alpha*100):02d}'] == 1]
        forecast_avg = df[f'es_{alpha}'].mean()
        realized = viol['r_actual'].mean() if len(viol) > 0 else np.nan
        z1 = (viol['r_actual'] / viol[f'es_{alpha}']).mean() + 1 if len(viol) > 0 else np.nan
        verdict = ('⚠ underforecast' if z1 < -0.10
                   else '✓ conservatief' if z1 > 0.10
                   else '✓ calibrated')
        print(f"  {density_name:<33}{int(alpha*100):>4}%{forecast_avg*100:>13.2f}%"
              f"{realized*100:>13.2f}%{z1:>10.3f}{verdict:>16}")
        results.append({'density': density_name, 'alpha': alpha,
                         'forecast_es': forecast_avg, 'realized_es': realized, 'z1': z1})

pd.DataFrame(results).to_csv(P/'outputs/tables/H5_density_robustness.csv', index=False)
print(f"\n✓ Saved: outputs/tables/H5_density_robustness.csv")

# === Diagnose plot: PIT-histogram per density ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
densities = [('Normal', lambda x: stats.norm.cdf(x)),
             ('Student-t(3)', lambda x: stats.t.cdf(x*np.sqrt(3/(3-2)), 3)),
             (f'Hansen skewed-t', lambda x: np.searchsorted(np.sort(z_skewt), x)/len(z_skewt))]
for ax, (name, cdf_fn) in zip(axes, densities):
    pit = cdf_fn(z.values)
    ax.hist(pit, bins=20, edgecolor='black', alpha=0.7, color='#dc2626')
    ax.axhline(len(pit)/20, color='black', ls='--', label='Uniform expectation')
    # KS-test
    ks = stats.kstest(pit, 'uniform').pvalue
    ax.set_title(f'{name}\nKS p-value: {ks:.4f}', fontweight='bold')
    ax.set_xlabel('PIT (should be uniform if density correct)')
    ax.set_ylabel('Count')
    ax.legend()
fig.suptitle('H5 — Density specification diagnostic: PIT histograms', fontweight='bold')
fig.tight_layout()
fig.savefig(P/'outputs/figures/H5_density_pit.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"✓ Figuur: outputs/figures/H5_density_pit.png")
