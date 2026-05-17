"""E2b — Christoffersen (1998) VaR coverage tests.

Drie likelihood-ratio tests voor VaR forecasts:
  LR_UC  — Unconditional Coverage: violation rate = α?
  LR_ind — Independence: zijn violations onafhankelijk over tijd?
  LR_cc  — Conditional Coverage: joint test van UC + ind
  
Reference: Christoffersen, P. F. (1998). Evaluating interval forecasts.
           International Economic Review, 39(4), 841-862.

Applied to HAR-RS-DOW return VaR forecasts at α ∈ {1%, 5%}.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

def lr_uc(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    """Unconditional Coverage test (Kupiec 1995).
    
    H0: P(viol) = alpha
    LR_UC = -2 * log(L(alpha) / L(pi_hat))
    onder H0: LR_UC ~ chi2(1)
    
    Returns: (LR_UC, p_value)
    """
    n = len(violations)
    x = int(violations.sum())  # observed violations
    pi_hat = x / n if n > 0 else 0
    
    if pi_hat == 0 or pi_hat == 1:
        return (0.0, 1.0)
    
    # Likelihood under H0
    L0 = (1 - alpha) ** (n - x) * alpha ** x
    # Likelihood under unrestricted
    L1 = (1 - pi_hat) ** (n - x) * pi_hat ** x
    
    if L0 <= 0 or L1 <= 0:
        return (np.nan, np.nan)
    
    lr = -2 * (np.log(L0) - np.log(L1))
    p = 1 - stats.chi2.cdf(lr, df=1)
    return (lr, p)

def lr_ind(violations: np.ndarray) -> tuple[float, float]:
    """Independence test (Christoffersen 1998).
    
    H0: violations are iid (no clustering)
    Markov chain transition probabilities:
      pi_01 = P(viol_t=1 | viol_{t-1}=0)
      pi_11 = P(viol_t=1 | viol_{t-1}=1)
    Under H0: pi_01 = pi_11 = pi
    
    Returns: (LR_ind, p_value)
    """
    v = violations.astype(int)
    n = len(v)
    if n < 2:
        return (np.nan, np.nan)
    
    # Transition counts
    n00 = ((v[:-1] == 0) & (v[1:] == 0)).sum()
    n01 = ((v[:-1] == 0) & (v[1:] == 1)).sum()
    n10 = ((v[:-1] == 1) & (v[1:] == 0)).sum()
    n11 = ((v[:-1] == 1) & (v[1:] == 1)).sum()
    
    # Transition probabilities  
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Avoid log(0)
    eps = 1e-10
    pi_01 = max(eps, min(1 - eps, pi_01))
    pi_11 = max(eps, min(1 - eps, pi_11))
    pi    = max(eps, min(1 - eps, pi))
    
    # Likelihoods
    log_L0 = n00*np.log(1-pi) + (n01+n11)*np.log(pi) + n10*np.log(1-pi)
    log_L1 = n00*np.log(1-pi_01) + n01*np.log(pi_01) + n10*np.log(1-pi_11) + n11*np.log(pi_11)
    
    lr = -2 * (log_L0 - log_L1)
    p = 1 - stats.chi2.cdf(lr, df=1)
    return (lr, p)

def lr_cc(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    """Conditional Coverage = LR_UC + LR_ind, chi2(2)"""
    lr_uc_, _ = lr_uc(violations, alpha)
    lr_ind_, _ = lr_ind(violations)
    if np.isnan(lr_uc_) or np.isnan(lr_ind_):
        return (np.nan, np.nan)
    lr_total = lr_uc_ + lr_ind_
    p = 1 - stats.chi2.cdf(lr_total, df=2)
    return (lr_total, p)

# === Load data en run tests voor HAR-RS-DOW ===
df = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
print(f"Sample: n = {len(df)} OOS days")
print(f"Period: {df.index[0]} → {df.index[-1]}")
print()

# Test voor 4 quantile-niveaus
print("="*90)
print("Christoffersen (1998) VaR coverage tests — HAR-RS-DOW return forecasts")
print("="*90)
print(f"{'Quantile':<10}{'Expected':<12}{'Observed':<12}{'LR_UC':<14}{'LR_ind':<14}{'LR_cc':<14}{'Verdict':<12}")
print("-"*90)

results = []
for col, alpha_nominal, name in [
    ('viol_return_01', 0.01, 'VaR 1% (left tail)'),
    ('viol_return_05', 0.05, 'VaR 5% (left tail)'),
    ('viol_return_95', 0.05, 'VaR 95% (right tail)'),
    ('viol_return_99', 0.01, 'VaR 99% (right tail)'),
]:
    v = df[col].values
    obs_rate = v.mean()
    lr_uc_val, p_uc = lr_uc(v, alpha_nominal)
    lr_ind_val, p_ind = lr_ind(v)
    lr_cc_val, p_cc = lr_cc(v, alpha_nominal)
    
    # Verdict: pass als alle p-values > 0.05
    pass_uc = '✓' if p_uc > 0.05 else '✗'
    pass_ind = '✓' if p_ind > 0.05 else '✗'
    pass_cc = '✓' if p_cc > 0.05 else '✗'
    verdict = 'PASS' if (p_uc > 0.05 and p_ind > 0.05 and p_cc > 0.05) else 'reject'
    
    results.append({
        'quantile': name, 'alpha_nominal': alpha_nominal,
        'expected_rate': alpha_nominal, 'observed_rate': obs_rate,
        'LR_UC': lr_uc_val, 'p_UC': p_uc,
        'LR_ind': lr_ind_val, 'p_ind': p_ind,
        'LR_CC': lr_cc_val, 'p_CC': p_cc,
    })
    
    print(f"{name:<22}{alpha_nominal*100:>6.1f}%{obs_rate*100:>10.2f}%   "
          f"{lr_uc_val:>6.2f}{pass_uc:>3} (p={p_uc:.3f}) "
          f"{lr_ind_val:>6.2f}{pass_ind:>3} (p={p_ind:.3f}) "
          f"{lr_cc_val:>6.2f}{pass_cc:>3} (p={p_cc:.3f})  {verdict}")

print()
print("Interpretatie:")
print("  LR_UC  > 5.99 (chi2_1, p<0.05): violation rate wijkt af van nominaal")
print("  LR_ind > 5.99 (chi2_1, p<0.05): violations clusteren (afhankelijk)")
print("  LR_cc  > 5.99 (chi2_2, p<0.05): joint test reject")
print("  ✓ = pass (p > 0.05), correct kalibratie")
print()

# Vergelijk met andere modellen (HAR-WE als alternatief)
print("="*90)
print("Vergelijking met HAR-WE (next-best uit horse race)")
print("="*90)
df_alt = pd.read_parquet(P/'outputs/tables/E1_walkforward_per_day_HAR-WE.parquet')
print(f"{'Model':<15}{'α=1%':<25}{'α=5%':<25}{'α=95%':<25}{'α=99%':<25}")
print("-"*100)
for label, dfi in [('HAR-RS-DOW', df), ('HAR-WE', df_alt)]:
    row = f"{label:<15}"
    for col, alpha in [('viol_return_01', 0.01), ('viol_return_05', 0.05),
                         ('viol_return_95', 0.05), ('viol_return_99', 0.01)]:
        v = dfi[col].values
        _, p_cc = lr_cc(v, alpha)
        symb = '✓' if p_cc > 0.05 else '✗'
        row += f"  p_CC={p_cc:.3f}{symb} ({v.mean()*100:.2f}%)"
    print(row)

# Save
pd.DataFrame(results).to_csv(P/'outputs/tables/E2b_christoffersen_var.csv', index=False)
print(f"\n✓ Saved: outputs/tables/E2b_christoffersen_var.csv")
