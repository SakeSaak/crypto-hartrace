"""HARState — the conditioning state ℱ_t for one-step-ahead forecasting."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


N_HIST = 22   # buffer length for monthly aggregate


@dataclass
class HARState:
    """Time-t state for HAR-family forecasting.

    Each history buffer stores the most recent ≥22 daily values, with
    the most recent observation at index -1.

    Attributes
    ----------
    log_RV : np.ndarray, shape (N_HIST,)
        History of log realized variance, most recent last.
    log_C, log_J : np.ndarray
        Continuous and jump components (Model A); log_J stores log(1+J).
    log_RS_neg, log_RS_pos : np.ndarray
        Negative and positive semi-variances (Model B).
    Q : float
        Measurement-error proxy at time t (= √RQ_t/RV_t minus sample mean).
    V, S, dD, F : float
        Exogenous regressors: demeaned log volume, log spread,
        ΔBTC-dominance, log USD/EUR FX-vol.
    sigma_prev : float
        σ_{t-1} for the scale recursion.
    eps_prev : float
        Standardized residual at t-1, for EGARCH/GAS update.
    """
    log_RV: np.ndarray
    log_C: np.ndarray
    log_J: np.ndarray
    log_RS_neg: np.ndarray
    log_RS_pos: np.ndarray
    Q: float
    V: float
    S: float
    dD: float
    F: float
    sigma_prev: float = 0.5
    eps_prev: float = 0.0

    # --- Cascade aggregates (D / W / M) ---
    @property
    def log_RV_D(self) -> float:
        return float(self.log_RV[-1])

    @property
    def log_RV_W(self) -> float:
        return float(np.mean(self.log_RV[-5:]))

    @property
    def log_RV_M(self) -> float:
        return float(np.mean(self.log_RV[-22:]))

    @property
    def log_C_D(self) -> float:
        return float(self.log_C[-1])

    @property
    def log_C_W(self) -> float:
        return float(np.mean(self.log_C[-5:]))

    @property
    def log_C_M(self) -> float:
        return float(np.mean(self.log_C[-22:]))

    @property
    def log_J_D(self) -> float:
        return float(self.log_J[-1])

    @property
    def log_J_W(self) -> float:
        return float(np.mean(self.log_J[-5:]))

    @property
    def log_J_M(self) -> float:
        return float(np.mean(self.log_J[-22:]))

    @property
    def log_RS_neg_D(self) -> float:
        return float(self.log_RS_neg[-1])

    @property
    def log_RS_pos_D(self) -> float:
        return float(self.log_RS_pos[-1])

    def copy(self) -> 'HARState':
        """Deep copy for independent MC paths."""
        return HARState(
            log_RV=self.log_RV.copy(),
            log_C=self.log_C.copy(),
            log_J=self.log_J.copy(),
            log_RS_neg=self.log_RS_neg.copy(),
            log_RS_pos=self.log_RS_pos.copy(),
            Q=self.Q, V=self.V, S=self.S, dD=self.dD, F=self.F,
            sigma_prev=self.sigma_prev, eps_prev=self.eps_prev,
        )
