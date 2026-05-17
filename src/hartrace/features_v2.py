"""Feature builder for extended HAR model families (v2).

Builds the full design-matrix-ready feature frame combining:
    - v2 realized measures   (RV, RK, BPV, RS+, RS-, J, etc.)
    - Periodicity            (weekend dummy + day-of-week dummies)
    - FX                     (EUR/USD daily return + 5-day rolling vol)
    - Macro                  (FRED: DFF, DGS10, T10Y2Y, VIX, USD index)

Target by default is log RK_{t+1} (noise-robust); pass target='log_RV'
to use simple-RV instead.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def build_features_v2(
    rv_path: str | Path,
    eurusd_path: Optional[str | Path] = None,
    fred_path: Optional[str | Path] = None,
    target: str = 'log_RK',
) -> pd.DataFrame:
    """Build a feature DataFrame ready for HAR-family OLS / MLE.

    Returns a DataFrame indexed by date with columns:
        target           = log RK_{t+1}  (or log_RV_{t+1})
        log_RK, log_RV, log_C, log_J, log_RS_neg, log_RS_pos
        Q, Q_neg, Q_pos    (HAR-Q interactions)
        *_D, *_W, *_M     (HAR cascade aggregates)
        is_weekend, dow_*  (periodicity dummies)
        eurusd_logret, eurusd_vol5
        macro_*           (lagged daily values)
    """
    df = pd.read_parquet(rv_path).copy()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)

    # Choose RV-target
    if target == 'log_RK':
        df['target_rv'] = df['RK']
    elif target == 'log_RV':
        df['target_rv'] = df['RV']
    else:
        raise ValueError(target)
    df['target'] = np.log(df['target_rv'].shift(-1))

    # Core log-transforms
    df['log_RK'] = np.log(df['RK'].clip(lower=1e-12))
    df['log_RV'] = np.log(df['RV'].clip(lower=1e-12))
    df['log_C']  = np.log(df['C'].clip(lower=1e-12))
    df['log_J']  = np.log1p(df['J'])
    df['log_RS_neg'] = np.log(df['RS_neg'].clip(lower=1e-12))
    df['log_RS_pos'] = np.log(df['RS_pos'].clip(lower=1e-12))

    # Use log_RK as the cascade base (noise-robust)
    for pref in ('log_RK', 'log_RV', 'log_C', 'log_J'):
        df[f'{pref}_D'] = df[pref]
        df[f'{pref}_W'] = df[pref].rolling(5).mean()
        df[f'{pref}_M'] = df[pref].rolling(22).mean()
    df['log_RS_neg_D'] = df['log_RS_neg']
    df['log_RS_pos_D'] = df['log_RS_pos']

    # HAR-Q proxy on the *kernel*-RV scale
    df['rq_proxy'] = np.sqrt(df['RQ']) / df['RK']
    df['Q'] = df['rq_proxy'] - df['rq_proxy'].mean()
    df['log_RK_D_Q']    = df['log_RK_D']    * df['Q']
    df['log_C_D_Q']     = df['log_C_D']     * df['Q']
    df['log_RS_neg_D_Q'] = df['log_RS_neg_D'] * df['Q']
    df['log_RS_pos_D_Q'] = df['log_RS_pos_D'] * df['Q']

    # Periodicity (already in v2 parquet: dow + is_weekend)
    if 'dow' not in df.columns:
        df['dow'] = df['date'].dt.dayofweek
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
    # DOW dummies (drop Mon=0 as baseline)
    for d in (1, 2, 3, 4, 5, 6):
        df[f'dow_{d}'] = (df['dow'] == d).astype(int)

    # FX exogenous (lagged so it's known at t when predicting t+1)
    if eurusd_path is not None and Path(eurusd_path).exists():
        fx = pd.read_parquet(eurusd_path)
        fx['date'] = pd.to_datetime(fx['date']).dt.tz_localize(None) if fx['date'].dt.tz is not None else pd.to_datetime(fx['date'])
        fx = fx.sort_values('date').reset_index(drop=True)
        fx['eurusd_vol5'] = fx['eurusd_logret'].rolling(5).std()
        df = df.merge(fx[['date', 'eurusd_logret', 'eurusd_vol5']],
                       on='date', how='left')
        df['eurusd_logret'] = df['eurusd_logret'].fillna(method='ffill')
        df['eurusd_vol5']   = df['eurusd_vol5'].fillna(method='ffill')
    else:
        df['eurusd_logret'] = 0.0
        df['eurusd_vol5']   = 0.0

    # Macro (lagged daily; FRED has business-day gaps so forward-fill is OK)
    if fred_path is not None and Path(fred_path).exists():
        fred = pd.read_parquet(fred_path)
        fred['date'] = pd.to_datetime(fred['date']).dt.tz_localize(None) if fred['date'].dt.tz is not None else pd.to_datetime(fred['date'])
        macro_cols = [c for c in fred.columns if c != 'date']
        fred = fred.sort_values('date').reset_index(drop=True)
        # forward-fill (carry last observation into weekends/holidays)
        fred[macro_cols] = fred[macro_cols].fillna(method='ffill')
        df = df.merge(fred[['date'] + macro_cols], on='date', how='left')
        for c in macro_cols:
            df[c] = df[c].fillna(method='ffill')
            df[f'macro_{c}'] = df[c]
        # Add a daily macro change (e.g. dDGS10) as a "news" regressor
        df['macro_dDGS10'] = df.get('DGS10', pd.Series(np.zeros(len(df)))).diff()
        df['macro_dVIX']   = df.get('VIXCLS', pd.Series(np.zeros(len(df)))).diff()
    else:
        for nm in ('macro_DFF', 'macro_DGS10', 'macro_T10Y2Y',
                    'macro_VIXCLS', 'macro_DTWEXBGS',
                    'macro_dDGS10', 'macro_dVIX'):
            df[nm] = 0.0

    df = df.set_index('date')
    return df


# =====================================================================
# Family column-spec for the extended horse race
# =====================================================================
FAMILIES_V2: dict[str, List[str]] = {
    # Tier 1 — classical baselines on log_RK
    'HAR':           ['const', 'log_RK_D', 'log_RK_W', 'log_RK_M'],
    'HAR-CJ':        ['const', 'log_C_D', 'log_C_W', 'log_C_M',
                       'log_J_D', 'log_J_W', 'log_J_M'],
    'HAR-RS':        ['const', 'log_RS_neg_D', 'log_RS_pos_D',
                       'log_RK_W', 'log_RK_M'],
    # Tier 2 — Q-correction
    'HAR-Q':         ['const', 'log_RK_D', 'log_RK_D_Q',
                       'log_RK_W', 'log_RK_M'],
    'HAR-RS-Q':      ['const', 'log_RS_neg_D', 'log_RS_neg_D_Q',
                       'log_RS_pos_D', 'log_RS_pos_D_Q',
                       'log_RK_W', 'log_RK_M'],
    # Tier 3 — NEW: periodicity
    'HAR-WE':        ['const', 'log_RK_D', 'log_RK_W', 'log_RK_M',
                       'is_weekend'],
    'HAR-RS-WE':     ['const', 'log_RS_neg_D', 'log_RS_pos_D',
                       'log_RK_W', 'log_RK_M', 'is_weekend'],
    'HAR-RS-DOW':    ['const', 'log_RS_neg_D', 'log_RS_pos_D',
                       'log_RK_W', 'log_RK_M',
                       'dow_1', 'dow_2', 'dow_3', 'dow_4',
                       'dow_5', 'dow_6'],
    # Tier 4 — NEW: HAR-MIDAS (with FX + macro)
    'HAR-RS-X':      ['const', 'log_RS_neg_D', 'log_RS_pos_D',
                       'log_RK_W', 'log_RK_M', 'is_weekend',
                       'eurusd_vol5', 'macro_VIXCLS', 'macro_T10Y2Y'],
    # Tier 5 — OUR hybrid: HAR-RS-Q-WE-X with skewed-t innovations
    'HAR-RS-Q-WE-X': ['const', 'log_RS_neg_D', 'log_RS_neg_D_Q',
                       'log_RS_pos_D', 'log_RS_pos_D_Q',
                       'log_RK_W', 'log_RK_M', 'is_weekend',
                       'eurusd_vol5', 'macro_VIXCLS', 'macro_T10Y2Y'],
}
