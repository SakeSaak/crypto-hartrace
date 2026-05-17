"""Daily realized measures v2 — noise-robust + extended stylized-facts.

Differences from v1 (realized.py):
    + Realized Kernel (RK) included as primary noise-robust RV estimator
    + Realized skewness and kurtosis (Amaya et al. 2015)
    + Signed jump component (Patton-Sheppard style)
    + Computed on 5-min returns (M ≈ 288), not 15-min (M ≈ 96)

For BNS jump-decomposition the formulas use simple RV (consistent with
literature conventions). RK is reported as an additional column for HAR
model fitting (it's the more accurate target).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma
from scipy.stats import norm

from .rv_noise_robust import realized_kernel


# Constants
MU1 = float(np.sqrt(2.0 / np.pi))
MU43 = float((2.0 ** (2.0 / 3.0)) * _gamma(7.0 / 6.0) / _gamma(0.5))
MU1_INV2 = 1.0 / (MU1 ** 2)
MU43_INV3 = 1.0 / (MU43 ** 3)
BNS_THETA = (MU1 ** -4) + 2.0 * (MU1 ** -2) - 5.0


def daily_v2(
    r: np.ndarray,
    jump_alpha: float = 0.999,
    min_bars: int = 240,   # ≈83% of 288 5-min bars
) -> Optional[dict]:
    """All daily measures from one day's 5-min log-returns.

    Returns a dict with the v1 measures plus RK, realized_skew,
    realized_kurt, and signed_jump.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    M = int(r.size)
    if M < min_bars:
        return None

    sq = r * r
    rv = float(sq.sum())
    rk = realized_kernel(r)
    rq = (M / 3.0) * float((sq * sq).sum())

    abs_r = np.abs(r)
    bpv = MU1_INV2 * float((abs_r[1:] * abs_r[:-1]).sum())

    if M >= 3:
        a43 = abs_r ** (4.0 / 3.0)
        tq = M * MU43_INV3 * float((a43[2:] * a43[1:-1] * a43[:-2]).sum())
    else:
        tq = float('nan')

    neg = r < 0
    rs_neg = float(sq[neg].sum())
    rs_pos = float(sq[~neg].sum())

    # BNS jump test (simple-RV based, per literature convention)
    if M >= 3 and np.isfinite(tq) and tq > 0 and bpv > 0 and rv > 0:
        ratio_j = (rv - bpv) / rv
        denom = np.sqrt((BNS_THETA / M) * max(1.0, tq / (bpv ** 2)))
        Z = ratio_j / denom if denom > 0 else float('nan')
    else:
        Z = float('nan')

    threshold = norm.ppf(jump_alpha)
    J_flag = int(np.isfinite(Z) and Z > threshold)
    if J_flag and rv > bpv:
        C = float(bpv)
        J = float(rv - bpv)
    else:
        C = float(rv)
        J = 0.0

    # Realized skewness and kurtosis (Amaya, Christoffersen, Jacobs, Vasquez 2015)
    if rv > 0:
        realized_skew = float(np.sqrt(M) * (r ** 3).sum() / (rv ** 1.5))
        realized_kurt = float(M * (r ** 4).sum() / (rv ** 2))
    else:
        realized_skew = float('nan')
        realized_kurt = float('nan')

    # Signed jump (Patton-Sheppard 2015 style; positive ⇒ upward-bias)
    signed_jump = float(rs_pos - rs_neg)

    return {
        'RV': rv, 'RK': rk, 'BPV': bpv, 'RQ': rq, 'TQ': tq,
        'RS_neg': rs_neg, 'RS_pos': rs_pos,
        'Z': Z, 'J_flag': J_flag, 'C': C, 'J': J,
        'signed_jump': signed_jump,
        'realized_skew': realized_skew,
        'realized_kurt': realized_kurt,
        'n_bars': M,
    }


def compute_v2_from_candles(
    candles: pd.DataFrame,
    ts_col: str = 'ts',
    close_col: str = 'close',
    volume_col: str = 'volume',
    ts_unit: str = 'ms',
    jump_alpha: float = 0.999,
    min_bars: int = 240,
) -> pd.DataFrame:
    """Build the v2 daily realized-measures table from intraday candles."""
    df = candles[[ts_col, close_col, volume_col]].copy()
    df['ts_dt'] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True)
    df = df.sort_values('ts_dt').drop_duplicates('ts_dt').reset_index(drop=True)
    df['log_close'] = np.log(df[close_col].astype(float))
    df['r'] = df['log_close'].diff()
    df['date'] = df['ts_dt'].dt.floor('D')
    df = df.dropna(subset=['r'])

    rows = []
    for date, day_df in df.groupby('date'):
        m = daily_v2(day_df['r'].values,
                      jump_alpha=jump_alpha, min_bars=min_bars)
        if m is None:
            continue
        m['date'] = pd.Timestamp(date)
        m['r_d'] = float(day_df['r'].sum())
        m['close'] = float(day_df[close_col].iloc[-1])
        m['volume'] = float(day_df[volume_col].sum())
        # Derived periodicity columns useful for stylized-fact tests
        m['dow'] = int(pd.Timestamp(date).dayofweek)  # 0=Mon
        m['is_weekend'] = int(m['dow'] in (5, 6))
        rows.append(m)

    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    out = out.set_index('date')
    return out
