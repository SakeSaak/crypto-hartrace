"""Daily realized measures from intraday returns.

Computes the full BNS-family of jump-robust volatility measures used by
the HAR-CJ-X-Q and HAR-RS-X-Q model families:

    RV     Realized variance               Andersen-Bollerslev (1998)
    BPV    Bipower variation               Barndorff-Nielsen & Shephard (2004)
    RQ     Realized quarticity             BNS (2002); used in HAR-Q
    TQ     Tripower quarticity             BNS (2006); jump-robust IQ
    RS-,RS+  Realized semi-variance        BN-Kinnebrock-Shephard (2010)
    Z       BNS jump test statistic         BNS (2006); Huang-Tauchen (2005)
    J       Jump component (truncated)     Andersen-Bollerslev-Diebold (2007)
    C       Continuous component           ABD (2007)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma
from scipy.stats import norm


# =====================================================================
# Constants
# =====================================================================
# μ_p = E|Z|^p voor Z ~ N(0,1) = 2^(p/2) Γ((p+1)/2) / Γ(1/2)
MU1 = float(np.sqrt(2.0 / np.pi))                        # μ_1     ≈ 0.7979
MU43 = float((2.0 ** (2.0 / 3.0)) * _gamma(7.0 / 6.0)    # μ_{4/3}
             / _gamma(0.5))                              #          ≈ 0.8309
MU1_INV2 = 1.0 / (MU1 ** 2)                              # μ_1^{-2} ≈ 1.5708
MU43_INV3 = 1.0 / (MU43 ** 3)                            # μ_{4/3}^{-3}

# BNS-test constant: μ_1^{-4} + 2 μ_1^{-2} - 5
BNS_THETA = (MU1 ** -4) + 2.0 * (MU1 ** -2) - 5.0        # ≈ 0.6090


# =====================================================================
# Per-day computation
# =====================================================================
def daily_measures(
    r: np.ndarray,
    jump_alpha: float = 0.999,
    min_bars: int = 80,
) -> Optional[dict]:
    """Compute all realized measures from intraday returns of one UTC day.

    Parameters
    ----------
    r : array-like of float
        Intraday log-returns (e.g., from 15-min closes), one UTC day.
    jump_alpha : float, default 0.999
        Probability threshold for BNS jump classification (one-sided).
        ABD (2007) recommend 0.999 for conservative jump detection.
    min_bars : int, default 80
        Minimum number of valid bars required (≈83% of 96 for 15-min).

    Returns
    -------
    dict or None
        Keys: RV, BPV, RQ, TQ, RS_neg, RS_pos, Z, J_flag, C, J, n_bars.
        Returns None if the day has too few bars.

    Notes
    -----
    The BNS jump test uses:
        Z = ((RV-BPV)/RV) / √( (BNS_θ / M) · max(1, TQ/BPV²) )
    Under H₀ of no jumps and M→∞, Z → N(0,1). Reject for Z > Φ⁻¹(α).

    The continuous/jump decomposition is the ABD (2007) significance-
    truncated version:
        J_t = 1{Z_t > Φ⁻¹(α)} · (RV_t - BPV_t)₊
        C_t = RV_t - J_t.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    M = int(r.size)

    if M < min_bars:
        return None

    # --- Basic returns-of-squares measures ---
    sq = r * r
    RV = float(sq.sum())
    RQ = (M / 3.0) * float((sq * sq).sum())

    # --- Bipower variation ---
    abs_r = np.abs(r)
    BPV = MU1_INV2 * float((abs_r[1:] * abs_r[:-1]).sum())

    # --- Tripower quarticity (jump-robust IQ proxy) ---
    if M >= 3:
        a43 = abs_r ** (4.0 / 3.0)
        TQ = M * MU43_INV3 * float((a43[2:] * a43[1:-1] * a43[:-2]).sum())
    else:
        TQ = float('nan')

    # --- Realized semi-variances ---
    neg = r < 0
    RS_neg = float(sq[neg].sum())
    RS_pos = float(sq[~neg].sum())

    # --- BNS jump test ---
    if (
        M >= 3 and np.isfinite(TQ) and TQ > 0.0 and BPV > 0.0 and RV > 0.0
    ):
        ratio_J = (RV - BPV) / RV
        denom = np.sqrt((BNS_THETA / M) * max(1.0, TQ / (BPV ** 2)))
        Z = ratio_J / denom if denom > 0 else float('nan')
    else:
        Z = float('nan')

    threshold = norm.ppf(jump_alpha)
    J_flag = int(np.isfinite(Z) and Z > threshold)

    # --- Continuous + jump decomposition (ABD 2007) ---
    if J_flag and RV > BPV:
        C = float(BPV)
        J = float(RV - BPV)
    else:
        C = float(RV)
        J = 0.0

    return {
        'RV': RV, 'BPV': BPV, 'RQ': RQ, 'TQ': TQ,
        'RS_neg': RS_neg, 'RS_pos': RS_pos,
        'Z': Z, 'J_flag': J_flag,
        'C': C, 'J': J,
        'n_bars': M,
    }


# =====================================================================
# Bar-level pipeline
# =====================================================================
def compute_from_candles(
    candles: pd.DataFrame,
    ts_col: str = 'ts',
    close_col: str = 'close',
    ts_unit: str = 'ms',
    jump_alpha: float = 0.999,
    min_bars: int = 80,
) -> pd.DataFrame:
    """Build the daily realized-measures table from a candles DataFrame.

    Parameters
    ----------
    candles : pd.DataFrame
        Must contain a timestamp column and a close-price column.
    ts_col, close_col : str
    ts_unit : {'ms', 's'}
        Unit of the integer timestamp column.
    jump_alpha : float
        Pass-through to `daily_measures`.
    min_bars : int
        Minimum bars per day to keep that day in the output.

    Returns
    -------
    pd.DataFrame indexed by UTC date with columns from `daily_measures`
    plus 'r_d' (daily log-return) and 'close' (last close of the day).
    """
    df = candles[[ts_col, close_col]].copy()
    df['ts_dt'] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True)
    df = df.sort_values('ts_dt').drop_duplicates('ts_dt').reset_index(drop=True)
    df['log_close'] = np.log(df[close_col].astype(float))
    df['r'] = df['log_close'].diff()
    df['date'] = df['ts_dt'].dt.floor('D')
    df = df.dropna(subset=['r'])

    rows = []
    for date, day_df in df.groupby('date'):
        meas = daily_measures(
            day_df['r'].values,
            jump_alpha=jump_alpha,
            min_bars=min_bars,
        )
        if meas is None:
            continue
        meas['date'] = pd.Timestamp(date)
        meas['r_d'] = float(day_df['r'].sum())
        meas['close'] = float(day_df[close_col].iloc[-1])
        rows.append(meas)

    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    out = out.set_index('date')
    return out
