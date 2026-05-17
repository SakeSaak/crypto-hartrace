"""E5 — Berkowitz (2001) PIT density forecast test.

Tests of de complete density forecast correct gespecificeerd is door:
1. Compute PIT_t = F_t(r_t) under hypothesized density
2. Transform z_t = Phi^{-1}(PIT_t) — should be iid N(0,1) onder H0
3. Joint LR test:
     a. Mean test    : E[z_t] = 0
     b. Variance test: Var(z_t) = 1
     c. Independence : no AR(1) structure
   onder H0: alle drie joint chi2(3)

Reference: Berkowitz, J. (2001). Testing density forecasts, with applications
           to risk management. JBES, 19(4), 465-474.

Applied to HAR-RS-DOW return density forecasts under Hansen (1994) skewed-t.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/Users/sakesaakstra/Desktop/crypto_hartrace/src')))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from hartrace.distributions import cdf as skewed_t_cdf

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

def compute_pit_hansen_skt(r: np.ndarray, sigma: np.ndarray, 
                            nu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """PIT onder Hansen skewed-t density: r_t / sigma_t ~ Hansen-skt(nu, lam)."""
    z = r / sigma
    pit = np.array([skewed_t_cdf(z[i], nu[i], lam[i]) for i in range(len(z))])
    # Clip om Phi^{-1}(0) en Phi^{-1}(1) te voorkomen
    pit = np.clip(pit, 1e-6, 1 - 1e-6)
    return pit

def compute_pit_normal(r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """PIT onder Normal density: r_t ~ N(0, sigma_t²)."""
    pit = stats.norm.cdf(r / sigma)
    return np.clip(pit, 1e-6, 1 - 1e-6)

def compute_pit_student_t(r: np.ndarray, sigma: np.ndarray, nu: float = 3.0) -> np.ndarray:
    """PIT onder standardized Student-t(nu) density."""
    # Standardised: scale factor sqrt(nu/(nu-2)) zodat var=1
    z = r / sigma * np.sqrt((nu - 2) / nu)
    pit = stats.t.cdf(z, df=nu)
    return np.clip(pit, 1e-6, 1 - 1e-6)

def berkowitz_lr_test(pit: np.ndarray) -> dict:
    """Berkowitz (2001) joint LR test op AR(1) model van transformed PITs.
    
    Model: z_t = mu + rho * z_{t-1} + epsilon_t, eps ~ N(0, sigma_z²)
    H0: mu=0, rho=0, sigma_z²=1
    """
    z = stats.norm.ppf(pit)
    n = len(z)
    
    # Unrestricted MLE
    def neg_loglik_unrestr(params):
        mu, rho, log_sig = params
        sig2 = np.exp(2*log_sig)
        # AR(1) likelihood, conditioneren op eerste obs
        eps = z[1:] - mu - rho * z[:-1]
        ll = -0.5 * (n-1) * (np.log(2*np.pi) + np.log(sig2)) - 0.5 * (eps**2).sum() / sig2
        return -ll
    
    res = minimize(neg_loglik_unrestr, [0, 0, 0], method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000})
    log_L1 = -res.fun
    mu_hat, rho_hat, log_sig_hat = res.x
    sig_hat = np.exp(log_sig_hat)
    
    # Restricted: mu=0, rho=0, sigma=1
    eps_r = z[1:] - 0 - 0 * z[:-1]
    log_L0 = -0.5 * (n-1) * np.log(2*np.pi) - 0.5 * (eps_r**2).sum()
    
    # LR
    lr = -2 * (log_L0 - log_L1)
    p_value = 1 - stats.chi2.cdf(lr, df=3)
    
    return {
        'lr_stat': lr, 'p_value': p_value, 'df': 3,
        'mu_hat': mu_hat, 'rho_hat': rho_hat, 'sigma_hat': sig_hat,
        'mean_z': z.mean(), 'std_z': z.std(),
        'pit_mean': pit.mean(), 'pit_std': pit.std(),
    }


# === Load data ===
df = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
print(f"Sample: n = {len(df)} OOS days")
print(f"Period: {df.index[0]} → {df.index[-1]}")
print()

# Compute σ_t = sqrt(E[RK_t | F_{t-1}]) = sqrt(exp(mu_pred + 0.5*sigma_pred²))
sigma_t = np.sqrt(np.exp(df['mu_pred'].values + 0.5 * df['sigma_pred'].values**2))
r_t = df['r_actual'].values

# === Test 1: PIT onder Hansen skewed-t (geestimeerde nu, lam per dag) ===
print("="*90)
print("Berkowitz (2001) PIT test — drie density specifications")
print("="*90)

results = []
for name, pit in [
    ('Normal',                compute_pit_normal(r_t, sigma_t)),
    ('Student-t (ν=3)',       compute_pit_student_t(r_t, sigma_t, nu=3.0)),
    ('Hansen skewed-t',       compute_pit_hansen_skt(r_t, sigma_t, df['nu_pred'].values, df['lam_pred'].values)),
]:
    r = berkowitz_lr_test(pit)
    verdict = '✓ pass' if r['p_value'] > 0.05 else '✗ reject'
    print(f"\n{name}:")
    print(f"  PIT-mean: {r['pit_mean']:.3f} (target=0.500)")
    print(f"  PIT-std:  {r['pit_std']:.3f} (target=0.289)")
    print(f"  z-mean:   {r['mean_z']:+.3f} (target=0.0)")
    print(f"  z-std:    {r['std_z']:.3f} (target=1.0)")
    print(f"  AR(1) ρ:  {r['rho_hat']:+.3f} (target=0.0)")
    print(f"  LR stat:  {r['lr_stat']:.2f}, df=3, p={r['p_value']:.4f}  →  {verdict}")
    results.append({'density': name, **r})

print()
print("="*90)
print("Interpretatie")
print("="*90)
print("""
Berkowitz LR test joint H0:
  - mu = 0    (location)
  - sigma = 1 (scale)
  - rho = 0   (no autocorrelation)
LR ~ chi²(3) onder H0. Reject (p<0.05) → density mis-specified.

PIT-mean ≠ 0.5  →  location bias (too low/high)
PIT-std  ≠ 0.289 →  density too wide / narrow (variance forecasting issue)
AR(1) ρ ≠ 0    →  remaining dynamics niet captured door σ_t
""")

pd.DataFrame(results).to_csv(P/'outputs/tables/E5_berkowitz_pit.csv', index=False)
print(f"✓ Saved: outputs/tables/E5_berkowitz_pit.csv")
