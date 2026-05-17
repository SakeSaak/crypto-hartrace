"""Parameter dataclasses for HAR-family models.

A `ModelSpec` bundles four sub-objects:
    - HARMeanParams         : coefficients for the cascade + Q + exogenous regressors
    - ScaleParams           : parameters for σ_t dynamics (static / GAS / EGARCH)
    - InnovationParams      : Hansen skewed-t (ν, λ) for log-RV shocks
    - ReturnInnovationParams: Hansen skewed-t (ν_r, λ_r) for return shocks given RV

The `family` attribute selects which mean-equation variant is active.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


Family = Literal['HAR', 'HAR-CJ', 'HAR-CJ-X-Q', 'HAR-RS', 'HAR-RS-X-Q']
ScaleType = Literal['static', 'gas', 'egarch']


@dataclass
class HARMeanParams:
    """Coefficients of the HAR mean equation.

    Family-specific coefficients:
      Model A (HAR-CJ-X-Q): uses β_D, β_D_Q, β_W, β_M for C^(D,W,M),
                            β_J_D, β_J_W, β_J_M for J^(D,W,M)
      Model B (HAR-RS-X-Q): uses β_D_neg, β_D_neg_Q, β_D_pos, β_D_pos_Q
                            for RS^-_D and RS^+_D, plus β_W, β_M on total RV
    Exogenous coefficients γ_V, γ_S, γ_dD, γ_F active only for X-Q variants.
    """
    beta_0: float = 0.0
    # HAR cascade — used by Model A (on log C) and HAR-baseline (on log RV)
    beta_D: float = 0.40
    beta_D_Q: float = 0.0
    beta_W: float = 0.30
    beta_M: float = 0.20
    # Jump-CJ component (Model A)
    beta_J_D: float = 0.0
    beta_J_W: float = 0.0
    beta_J_M: float = 0.0
    # Semi-variance daily (Model B)
    beta_D_neg: float = 0.0
    beta_D_neg_Q: float = 0.0
    beta_D_pos: float = 0.0
    beta_D_pos_Q: float = 0.0
    # Exogenous regressors
    gamma_V: float = 0.0   # demeaned log volume
    gamma_S: float = 0.0   # log bid-ask spread
    gamma_dD: float = 0.0  # ΔBTC-dominance
    gamma_F: float = 0.0   # log USD/EUR FX-vol


@dataclass
class ScaleParams:
    """Conditional scale σ_t dynamics of HAR residuals."""
    scale_type: ScaleType = 'static'
    sigma_static: float = 0.50
    # GAS(1,1) on log σ²_t with skewed-t score
    omega_gas: float = 0.0
    A_gas: float = 0.05
    B_gas: float = 0.95
    # EGARCH(1,1) with leverage
    omega_egarch: float = 0.0
    alpha_egarch: float = 0.10
    delta_egarch: float = -0.05
    beta_egarch: float = 0.95


@dataclass
class InnovationParams:
    """Hansen skewed-t (ν, λ) for log-RV shocks ε_t."""
    nu: float = 8.0
    lam: float = -0.10


@dataclass
class ReturnInnovationParams:
    """Hansen skewed-t (ν_r, λ_r) for return shocks η_t given RV_t."""
    nu_r: float = 5.0
    lam_r: float = -0.15
    mu_r: float = 0.0  # mean return; near zero for crypto


@dataclass
class ModelSpec:
    """Complete fitted-model specification."""
    family: Family = 'HAR-CJ-X-Q'
    mean: HARMeanParams = field(default_factory=HARMeanParams)
    scale: ScaleParams = field(default_factory=ScaleParams)
    innov: InnovationParams = field(default_factory=InnovationParams)
    return_innov: ReturnInnovationParams = field(default_factory=ReturnInnovationParams)
