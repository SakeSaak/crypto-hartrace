"""Hansen (1994) skewed-t distribution.

Standardized (mean 0, variance 1) skewed-t density with degrees of
freedom ν > 2 and skewness λ ∈ (-1, 1). At λ = 0, reduces to the
standardized Student-t. Used as the innovation distribution for the
log-RV shocks in HAR-CJ-X-Q-skT and HAR-RS-X-Q-skT.

References
----------
Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation",
International Economic Review 35, 705-730.
"""

from __future__ import annotations
from typing import Tuple, Union

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t


# ---------------------------------------------------------------------
# Hansen's a, b, c constants
# ---------------------------------------------------------------------
def _abc(nu: float, lam: float) -> Tuple[float, float, float]:
    """Compute Hansen's normalising constants.

    c = Γ((ν+1)/2) / (√(π(ν-2)) Γ(ν/2))
    a = 4 λ c (ν-2)/(ν-1)
    b² = 1 + 3 λ² − a²
    """
    if nu <= 2.0:
        raise ValueError(f"nu must be > 2 for finite variance (got {nu})")
    if not -1.0 < lam < 1.0:
        raise ValueError(f"lam must be in (-1, 1) (got {lam})")
    log_c = (
        gammaln(0.5 * (nu + 1.0))
        - 0.5 * np.log(np.pi * (nu - 2.0))
        - gammaln(0.5 * nu)
    )
    c = float(np.exp(log_c))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = float(np.sqrt(1.0 + 3.0 * lam ** 2 - a ** 2))
    return a, b, c


# ---------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------
def pdf(z: Union[float, np.ndarray], nu: float, lam: float) -> np.ndarray:
    """Density of standardized Hansen skewed-t."""
    a, b, c = _abc(nu, lam)
    z = np.asarray(z, dtype=float)
    left = z < -a / b
    denom = np.where(left, 1.0 - lam, 1.0 + lam)
    quad = ((b * z + a) / denom) ** 2 / (nu - 2.0)
    return b * c * (1.0 + quad) ** (-0.5 * (nu + 1.0))


def loglik(z: Union[float, np.ndarray], nu: float, lam: float) -> np.ndarray:
    """Log-density of standardized Hansen skewed-t."""
    a, b, c = _abc(nu, lam)
    z = np.asarray(z, dtype=float)
    left = z < -a / b
    denom = np.where(left, 1.0 - lam, 1.0 + lam)
    quad = ((b * z + a) / denom) ** 2 / (nu - 2.0)
    log_c = (
        gammaln(0.5 * (nu + 1.0))
        - 0.5 * np.log(np.pi * (nu - 2.0))
        - gammaln(0.5 * nu)
    )
    return np.log(b) + log_c - 0.5 * (nu + 1.0) * np.log1p(quad)


# ---------------------------------------------------------------------
# CDF — derivation:
#   Left  (z < -a/b):  F(z) = (1-λ) · G*((bz+a)/(1-λ))
#   Right (z ≥ -a/b):  F(z) = −λ + (1+λ) · G*((bz+a)/(1+λ))
# where G*(·) = CDF of the variance-1-scaled Student-t with df ν, which
# relates to the standard Student-t CDF as G*(x) = T_ν(x · √(ν/(ν-2))).
# ---------------------------------------------------------------------
def cdf(z: Union[float, np.ndarray], nu: float, lam: float) -> np.ndarray:
    """CDF of standardized Hansen skewed-t."""
    a, b, _ = _abc(nu, lam)
    z = np.asarray(z, dtype=float)
    left = z < -a / b
    scale_in = np.sqrt(nu / (nu - 2.0))  # variance-1 → standard t-scale
    out = np.empty_like(z, dtype=float)
    if np.any(left):
        zl = z[left]
        arg = scale_in * (b * zl + a) / (1.0 - lam)
        out[left] = (1.0 - lam) * student_t.cdf(arg, df=nu)
    if np.any(~left):
        zr = z[~left]
        arg = scale_in * (b * zr + a) / (1.0 + lam)
        out[~left] = -lam + (1.0 + lam) * student_t.cdf(arg, df=nu)
    return out


# ---------------------------------------------------------------------
# Inverse CDF — invert the two-branch form analytically.
# ---------------------------------------------------------------------
def ppf(u: Union[float, np.ndarray], nu: float, lam: float) -> np.ndarray:
    """Inverse CDF (quantile function) of standardized Hansen skewed-t."""
    a, b, _ = _abc(nu, lam)
    u = np.asarray(u, dtype=float)
    threshold = 0.5 * (1.0 - lam)
    left = u < threshold
    scale_out = np.sqrt((nu - 2.0) / nu)  # standard t → variance-1
    out = np.empty_like(u, dtype=float)
    if np.any(left):
        p_l = u[left] / (1.0 - lam)
        t_l = student_t.ppf(p_l, df=nu)
        out[left] = ((1.0 - lam) * scale_out * t_l - a) / b
    if np.any(~left):
        p_r = (u[~left] + lam) / (1.0 + lam)
        t_r = student_t.ppf(p_r, df=nu)
        out[~left] = ((1.0 + lam) * scale_out * t_r - a) / b
    return out


# ---------------------------------------------------------------------
# Random variates
# ---------------------------------------------------------------------
def rvs(
    nu: float,
    lam: float,
    size: Union[int, Tuple[int, ...]],
    rng: Union[np.random.Generator, None] = None,
) -> np.ndarray:
    """Draw samples from standardized Hansen skewed-t via inverse-CDF."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=size)
    return ppf(u, nu, lam)
