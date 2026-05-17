"""Multi-step Monte Carlo density forecaster for HAR-family models.

Translates the pseudocode of `simulate_density(θ̂, ℱ_t, h, K, AUX, COMP)`
into a Python implementation. Produces K paths of (log RV, return) over
horizon h, from which VaR_α, ES_α and density-forecast scores (CRPS,
log-score, PIT) are computed downstream.

The path loop is kept explicit and readable; vectorisation over K paths
is a Fase-2 optimisation once the skeleton is validated.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import distributions as skt
from .params import ModelSpec
from .state import HARState
from .dynamics import har_mean, update_sigma


# ---------------------------------------------------------------------
# Auxiliary AR(1) forecasters for exogenous regressors
# ---------------------------------------------------------------------
@dataclass
class AR1:
    """AR(1) for one exogenous regressor: x_{t+1} = c + ρ x_t + σ ε."""
    c: float = 0.0
    rho: float = 0.9
    sigma: float = 0.3

    def step(self, x_t: float, rng: np.random.Generator) -> float:
        return self.c + self.rho * x_t + self.sigma * rng.standard_normal()


@dataclass
class AuxForecasters:
    """Bundle of AR(1) forecasters for V, S, ΔD, F."""
    V: AR1 = None
    S: AR1 = None
    dD: AR1 = None
    F: AR1 = None

    def __post_init__(self):
        if self.V is None:
            self.V = AR1(c=0.0, rho=0.90, sigma=0.50)
        if self.S is None:
            self.S = AR1(c=0.0, rho=0.80, sigma=0.30)
        if self.dD is None:
            self.dD = AR1(c=0.0, rho=0.50, sigma=0.10)
        if self.F is None:
            self.F = AR1(c=0.0, rho=0.95, sigma=0.20)


# ---------------------------------------------------------------------
# Bootstrap pool of historical component shares
# ---------------------------------------------------------------------
@dataclass
class ComponentSampler:
    """Bootstrap pool of historical (component / RV) shares.

    Stores arrays of length N_hist, from which one row is drawn uniformly
    per simulation step to form the component decomposition of a
    simulated RV draw. Preserves the empirical joint distribution of
    (C/RV, J/RV, RS-/RV, RS+/RV, √RQ/RV).
    """
    shares_C: np.ndarray       # C_t / RV_t
    shares_J: np.ndarray       # J_t / RV_t (truncated ≥ 0)
    shares_RS_neg: np.ndarray  # RS-_t / RV_t
    shares_RS_pos: np.ndarray  # RS+_t / RV_t
    shares_RQ: np.ndarray      # demeaned √RQ_t / RV_t  (for Q_t update)

    def sample_indices(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.integers(0, len(self.shares_C), size=n)


# ---------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Output of `simulate(...)`.

    Attributes
    ----------
    paths_log_RV : np.ndarray, shape (K, h)
        Simulated log RV per path per horizon step.
    paths_return : np.ndarray, shape (K, h)
        Simulated daily returns per path per horizon step.
    cum_return : np.ndarray, shape (K,)
        Cumulative h-period return per path.
    horizon : int
    K : int
    """
    paths_log_RV: np.ndarray
    paths_return: np.ndarray
    cum_return: np.ndarray
    horizon: int
    K: int

    def var(self, alpha: float = 0.01) -> float:
        """Empirical VaR_α (positive number) of cumulative h-period return."""
        return float(-np.quantile(self.cum_return, alpha))

    def es(self, alpha: float = 0.01) -> float:
        """Empirical ES_α (positive number) of cumulative h-period return."""
        cutoff = np.quantile(self.cum_return, alpha)
        tail = self.cum_return[self.cum_return <= cutoff]
        if tail.size == 0:
            return float('nan')
        return float(-tail.mean())

    def crps(self, observed: float) -> float:
        """Empirical CRPS for one realised observation.

        CRPS(F, y) = E|X − y| − ½ E|X − X'|, with X, X' ~ F i.i.d.
        Estimated unbiasedly from the simulated cum_return sample.
        """
        x = self.cum_return
        K = len(x)
        # First term: mean |X - y|
        term1 = float(np.mean(np.abs(x - observed)))
        # Second term: ½ E|X - X'|. Sort + sum-of-differences trick is O(K log K).
        sx = np.sort(x)
        i = np.arange(1, K + 1)
        term2 = float(np.mean((2.0 * i - K - 1.0) * sx)) / K
        return term1 - 0.5 * term2

    def pit(self, observed: float) -> float:
        """Probability integral transform: F̂(y) for realised y."""
        return float(np.mean(self.cum_return <= observed))


# ---------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------
def _roll(history: np.ndarray, new_val: float) -> np.ndarray:
    """Roll a 1-D history buffer one step forward."""
    out = np.empty_like(history)
    out[:-1] = history[1:]
    out[-1] = new_val
    return out


def simulate(
    spec: ModelSpec,
    state_t: HARState,
    horizon: int,
    K: int,
    aux: AuxForecasters,
    components: ComponentSampler,
    rng: Optional[np.random.Generator] = None,
    truncate_jumps: bool = True,
) -> SimulationResult:
    """Simulate K density-forecast paths for horizons 1..h.

    Parameters
    ----------
    spec : ModelSpec
        Fitted model parameters (or mock for development).
    state_t : HARState
        Conditioning state at time t.
    horizon : int
        Forecast horizon h.
    K : int
        Number of MC paths.
    aux : AuxForecasters
        AR(1) forecasters for V, S, ΔD, F.
    components : ComponentSampler
        Bootstrap pool of historical component-share tuples.
    rng : np.random.Generator, optional
        Generator for reproducibility.
    truncate_jumps : bool
        If True, sampled J* values are floored at zero.

    Returns
    -------
    SimulationResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-sample all shocks (K × h matrix) — vectorised draw from skewed-t
    eps_RV = skt.rvs(spec.innov.nu, spec.innov.lam, size=(K, horizon), rng=rng)
    eta_r = skt.rvs(
        spec.return_innov.nu_r, spec.return_innov.lam_r,
        size=(K, horizon), rng=rng,
    )

    # Pre-sample bootstrap indices for component decomposition
    comp_idx = components.sample_indices(K * horizon, rng).reshape(K, horizon)

    paths_log_RV = np.empty((K, horizon))
    paths_return = np.empty((K, horizon))

    mu_r = spec.return_innov.mu_r

    for k in range(K):
        s = state_t.copy()
        sigma_t = s.sigma_prev

        for j in range(horizon):
            # (1) Mean
            mu = har_mean(s, spec)

            # (2) Shock & log-RV
            eps = eps_RV[k, j]
            log_RV_new = mu + sigma_t * eps
            RV_new = float(np.exp(log_RV_new))
            paths_log_RV[k, j] = log_RV_new

            # (3) Component decomposition via bootstrap
            ix = comp_idx[k, j]
            C_new = components.shares_C[ix] * RV_new
            J_share = components.shares_J[ix]
            J_new = max(J_share * RV_new, 0.0) if truncate_jumps else J_share * RV_new
            RS_neg_new = components.shares_RS_neg[ix] * RV_new
            RS_pos_new = components.shares_RS_pos[ix] * RV_new
            Q_new = float(components.shares_RQ[ix])  # already demeaned

            # (4) Exogenous AR(1) updates
            V_new = aux.V.step(s.V, rng)
            S_new = aux.S.step(s.S, rng)
            dD_new = aux.dD.step(s.dD, rng)
            F_new = aux.F.step(s.F, rng)

            # (5) Scale update
            sigma_next = update_sigma(sigma_t, eps, spec)

            # (6) Roll state forward
            s.log_RV = _roll(s.log_RV, log_RV_new)
            s.log_C = _roll(s.log_C, float(np.log(max(C_new, 1e-12))))
            s.log_J = _roll(s.log_J, float(np.log1p(J_new)))
            s.log_RS_neg = _roll(s.log_RS_neg, float(np.log(max(RS_neg_new, 1e-12))))
            s.log_RS_pos = _roll(s.log_RS_pos, float(np.log(max(RS_pos_new, 1e-12))))
            s.Q = Q_new
            s.V, s.S, s.dD, s.F = V_new, S_new, dD_new, F_new
            s.eps_prev = float(eps)
            sigma_t = sigma_next

            # (7) Return draw (vol-time-change assumption)
            paths_return[k, j] = mu_r + np.sqrt(RV_new) * eta_r[k, j]

    cum_return = paths_return.sum(axis=1)

    return SimulationResult(
        paths_log_RV=paths_log_RV,
        paths_return=paths_return,
        cum_return=cum_return,
        horizon=horizon,
        K=K,
    )
