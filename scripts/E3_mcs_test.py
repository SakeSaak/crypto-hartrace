"""E3 — Hansen, Lunde, Nason (2011) Model Confidence Set test.

Identificeert de "Set of Best Models" — alle modellen waarvan we niet kunnen
verwerpen dat ze equivalent zijn aan het beste model bij confidence level (1-α).

Algoritme:
  1. Start with set M = {alle modellen}
  2. T_R,M = max over (i,j) van t-statistiek voor verschil in mean losses
  3. Bootstrap voor p-value via stationary block bootstrap
  4. Reject equal predictive ability als p < α: elimineer worst model
  5. Repeat tot p >= α: huidige M = MCS bij confidence level (1-α)

Reference: Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model
           Confidence Set. Econometrica, 79(2), 453-497.

Applied to OOS CRPS losses for 5 HAR-family specifications.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

def stationary_bootstrap_sample(n: int, avg_block_len: float = 5.0) -> np.ndarray:
    """Politis-Romano stationary block bootstrap indices."""
    p = 1.0 / avg_block_len
    indices = np.zeros(n, dtype=int)
    indices[0] = np.random.randint(0, n)
    for t in range(1, n):
        if np.random.random() < p:
            indices[t] = np.random.randint(0, n)
        else:
            indices[t] = (indices[t-1] + 1) % n
    return indices

def mcs_test(losses: dict[str, np.ndarray], alpha: float = 0.10,
             n_bootstrap: int = 5000, avg_block: float = 5.0,
             rng_seed: int = 42) -> dict:
    """Hansen-Lunde-Nason MCS test op een dict van model_name → loss series.
    
    Returns: dict met:
      - mcs_models: lijst van modellen in MCS bij (1-alpha) confidence
      - eliminated: lijst van modellen geëlimineerd, met p-values
      - final_p: p-value bij final iteratie
    """
    np.random.seed(rng_seed)
    M = list(losses.keys())
    loss_matrix = np.column_stack([losses[m] for m in M])  # (n, k)
    n, k = loss_matrix.shape
    
    eliminated = []
    iteration = 0
    
    while len(M) > 1:
        iteration += 1
        # Compute mean losses voor huidige set
        mean_losses = {m: losses[m].mean() for m in M}
        worst = max(mean_losses, key=mean_losses.get)  # highest loss = slechtste model
        
        # Loss differentials: d_ij = L_i - L_j for all pairs
        cur_idx = [M.index(m) for m in M]
        cur_losses = loss_matrix[:, cur_idx] if len(M) < k else loss_matrix
        # Compute d_i_dot = avg over j of (L_i - L_j) (relative to set mean)
        L_bar = cur_losses.mean(axis=1, keepdims=True)
        d = cur_losses - L_bar  # relative loss
        
        # T_R statistic: max-pairwise t-stat
        T_pairs = []
        for i in range(len(M)):
            for j in range(i+1, len(M)):
                d_ij = cur_losses[:, i] - cur_losses[:, j]
                mean_d = d_ij.mean()
                # HAC variance via bootstrap
                var_d = d_ij.var(ddof=1) / n  # eenvoudige estimator
                if var_d > 0:
                    t_stat = abs(mean_d) / np.sqrt(var_d)
                    T_pairs.append(t_stat)
        T_obs = max(T_pairs) if T_pairs else 0.0
        
        # Bootstrap T-statistic distributie  
        T_boot = []
        for b in range(n_bootstrap):
            idx = stationary_bootstrap_sample(n, avg_block)
            L_b = cur_losses[idx]
            T_b_pairs = []
            for i in range(len(M)):
                for j in range(i+1, len(M)):
                    d_b = L_b[:, i] - L_b[:, j]
                    # Recenter via observed mean
                    d_b_obs = cur_losses[:, i] - cur_losses[:, j]
                    mean_d_b = d_b.mean() - d_b_obs.mean()  # recentered
                    var_d_b = d_b.var(ddof=1) / n
                    if var_d_b > 0:
                        T_b_pairs.append(abs(mean_d_b) / np.sqrt(var_d_b))
            if T_b_pairs:
                T_boot.append(max(T_b_pairs))
        
        T_boot = np.array(T_boot)
        p_value = (T_boot >= T_obs).mean()
        
        print(f"  Iter {iteration}: M={M}, T_obs={T_obs:.3f}, p={p_value:.4f}", end='')
        
        if p_value >= alpha:
            print(f"  → STOP (p ≥ {alpha}, MCS converged)")
            break
        else:
            print(f"  → eliminate {worst}")
            eliminated.append({'iteration': iteration, 'model': worst,
                               'p_value': p_value, 'T_obs': T_obs})
            M.remove(worst)
    
    return {
        'mcs_models': M,
        'eliminated': eliminated,
        'final_p': p_value if len(M) > 1 else 1.0,
        'iterations': iteration,
        'confidence_level': 1 - alpha,
    }


# === Load OOS losses voor alle 5 modellen ===
models_files = {
    'HAR-RS-DOW':     P/'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet',
    'HAR-RS-Q-WE-X':  P/'outputs/tables/E1_walkforward_per_day_HAR-RS-Q-WE-X.parquet',
    'HAR-RS-WE':      P/'outputs/tables/E1_walkforward_per_day_HAR-RS-WE.parquet',
    'HAR-RS-X':       P/'outputs/tables/E1_walkforward_per_day_HAR-RS-X.parquet',
    'HAR-WE':         P/'outputs/tables/E1_walkforward_per_day_HAR-WE.parquet',
}

crps_dict = {}
qlike_dict = {}
for name, fpath in models_files.items():
    df = pd.read_parquet(fpath)
    crps_dict[name] = df['crps'].dropna().values
    qlike_dict[name] = df['qlike'].dropna().values

# Verifieer alle modellen hebben gelijke lengte
n_lens = {name: len(v) for name, v in crps_dict.items()}
print(f"OOS sample sizes: {n_lens}")
print()

# === MCS tests ===
print("="*90)
print("Hansen-Lunde-Nason (2011) Model Confidence Set test on OOS losses")
print("="*90)
print()

print("Mean losses per model:")
print(f"{'Model':<20}{'Mean CRPS':<14}{'Mean QLIKE':<14}")
for name in models_files:
    print(f"  {name:<20}{crps_dict[name].mean():<14.4f}{qlike_dict[name].mean():<14.4f}")
print()

# Run MCS on CRPS at α=0.10 (90% confidence)
print("--- MCS test on CRPS, α=0.10 (90% confidence) ---")
result_crps_90 = mcs_test(crps_dict, alpha=0.10, n_bootstrap=2000)
print()
print(f"MCS (90% confidence): {result_crps_90['mcs_models']}")
print()

# Run MCS on CRPS at α=0.25 (75% confidence — strenger)
print("--- MCS test on CRPS, α=0.25 (75% confidence) ---")
result_crps_75 = mcs_test(crps_dict, alpha=0.25, n_bootstrap=2000)
print()
print(f"MCS (75% confidence): {result_crps_75['mcs_models']}")
print()

# Run MCS on QLIKE at α=0.10
print("--- MCS test on QLIKE, α=0.10 (90% confidence) ---")
result_qlike_90 = mcs_test(qlike_dict, alpha=0.10, n_bootstrap=2000)
print()
print(f"MCS (90% confidence, QLIKE): {result_qlike_90['mcs_models']}")
print()

# Save resultaten
results_df = pd.DataFrame([
    {'loss': 'CRPS', 'conf_level': 0.90, 'mcs': str(result_crps_90['mcs_models']),
     'eliminated': str([e['model'] for e in result_crps_90['eliminated']])},
    {'loss': 'CRPS', 'conf_level': 0.75, 'mcs': str(result_crps_75['mcs_models']),
     'eliminated': str([e['model'] for e in result_crps_75['eliminated']])},
    {'loss': 'QLIKE', 'conf_level': 0.90, 'mcs': str(result_qlike_90['mcs_models']),
     'eliminated': str([e['model'] for e in result_qlike_90['eliminated']])},
])
results_df.to_csv(P/'outputs/tables/E3_mcs_test.csv', index=False)
print(f"✓ Saved: outputs/tables/E3_mcs_test.csv")
print()

# === Interpretatie ===
print("="*90)
print("Interpretatie")
print("="*90)
print(f"""
Onder strengste eis (90% confidence, CRPS):
  MCS bestaat uit: {result_crps_90['mcs_models']}
  → {len(result_crps_90['mcs_models'])} model(en) overleven het stringent test

Als HAR-RS-DOW de ENIGE in MCS is, dan is het op formele basis (Hansen-Lunde-Nason
multiple-comparison framework) het uniek superieure model.

Diebold-Mariano paarsgewijze tests rejecten al equal predictive accuracy, maar
MCS controleert voor multiple testing — strenger en informatiever.
""")
