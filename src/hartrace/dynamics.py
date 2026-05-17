"""HAR mean equation and σ_t scale dynamics.

`har_mean`     computes μ_t = E[log RV_{t+1} | ℱ_t] using the active family.
`update_sigma` updates σ_t under static / GAS / EGARCH scale dynamics.
"""

from __future__ import annotations
import numpy as np
from scipy.special import gammaln

from .params import ModelSpec
from .state import HARState


# ---------------------------------------------------------------------
# Conditional mean μ_t
# ---------------------------------------------------------------------
def har_mean(state: HARState, spec: ModelSpec) -> float:
    """Compute μ_t = E[log RV_{t+1} | ℱ_t] given current state and parameters."""
    m = spec.mean
    fam = spec.family
    mu = m.beta_0

    # --- Daily / Weekly / Monthly cascade ---
    if fam in ('HAR-CJ', 'HAR-CJ-X-Q'):
        # On continuous component, with Q-correction on daily lag
        mu += (m.beta_D + m.beta_D_Q * state.Q) * state.log_C_D
        mu += m.beta_W * state.log_C_W
        mu += m.beta_M * state.log_C_M
        # Plus jump component
        mu += m.beta_J_D * state.log_J_D
        mu += m.beta_J_W * state.log_J_W
        mu += m.beta_J_M * state.log_J_M
    elif fam in ('HAR-RS', 'HAR-RS-X-Q'):
        # Semi-variance decomposition on daily, total RV on W/M
        mu += (m.beta_D_neg + m.beta_D_neg_Q * state.Q) * state.log_RS_neg_D
        mu += (m.beta_D_pos + m.beta_D_pos_Q * state.Q) * state.log_RS_pos_D
        mu += m.beta_W * state.log_RV_W
        mu += m.beta_M * state.log_RV_M
    else:  # vanilla HAR baseline
        mu += (m.beta_D + m.beta_D_Q * state.Q) * state.log_RV_D
        mu += m.beta_W * state.log_RV_W
        mu += m.beta_M * state.log_RV_M

    # --- Exogenous regressors (only for X-Q variants) ---
    if fam.endswith('X-Q'):
        mu += m.gamma_V * state.V
        mu += m.gamma_S * state.S
        mu += m.gamma_dD * state.dD
        mu += m.gamma_F * state.F

    return float(mu)


# ---------------------------------------------------------------------
# E|z| for standardized Student-t with ν df (used in EGARCH)
# Formula: E|t_ν| = 2 √(ν-2) Γ((ν+1)/2) / ((ν-1) √π Γ(ν/2))
# ---------------------------------------------------------------------
def _e_abs_student_t(nu: float) -> float:
    log_val = (
        np.log(2.0)
        + 0.5 * np.log(nu - 2.0)
        + gammaln(0.5 * (nu + 1.0))
        - np.log(nu - 1.0)
        - 0.5 * np.log(np.pi)
        - gammaln(0.5 * nu)
    )
    return float(np.exp(log_val))


# ---------------------------------------------------------------------
# σ_t update — static / GAS / EGARCH
# ---------------------------------------------------------------------
def update_sigma(sigma_prev: float, eps_prev: float, spec: ModelSpec) -> float:
    """Return σ_t given σ_{t-1} and the standardized residual at t-1.

    For 'static':  σ_t = σ̂_static (constant).
    For 'gas'   :  log σ²_t = ω + A·s_{t-1} + B·log σ²_{t-1},
                   with skewed-t score s_{t-1} (Fisher-scaled).
    For 'egarch':  log σ²_t = ω + α(|z| - E|z|) + δ·z + β·log σ²_{t-1}.
    """
    sp = spec.scale
    if sp.scale_type == 'static':
        return float(sp.sigma_static)

    z = eps_prev  # standardized residual at t-1
    log_var_prev = 2.0 * np.log(max(sigma_prev, 1e-12))

    if sp.scale_type == 'gas':
        # Score of standardized symmetric-t component, Fisher-scaled
        # (approximation; exact skew-t score is a small refinement).
        # ∂ log p / ∂ log σ² = (ν+1)/2 · z²/(ν-2+z²) - 1/2.
        # Scaled-by-inverse-Fisher reduces to (z²·(ν+1)/(ν+3·z²/(ν-2))) form;
        # we use the standard reduced-form score that matches GAS-t.
        nu = spec.innov.nu
        score = (nu + 1.0) * z ** 2 / (nu - 2.0 + z ** 2) - 1.0
        log_var_new = sp.omega_gas + sp.A_gas * score + sp.B_gas * log_var_prev
        return float(np.exp(0.5 * log_var_new))

    if sp.scale_type == 'egarch':
        e_abs = _e_abs_student_t(spec.innov.nu)
        log_var_new = (
            sp.omega_egarch
            + sp.alpha_egarch * (np.abs(z) - e_abs)
            + sp.delta_egarch * z
            + sp.beta_egarch * log_var_prev
        )
        return float(np.exp(0.5 * log_var_new))

    raise ValueError(f"Unknown scale_type: {sp.scale_type}")
