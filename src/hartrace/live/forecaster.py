"""Daily HAR-RS-Q-WE-X forecast — wrapper rond bestaande hartrace-pipeline.

Workflow:
  1. Load historical realized measures (saved daily)
  2. Append today's RV if not present
  3. Build features
  4. Use latest saved fit (refit if > REFIT_INTERVAL days oud)
  5. Compute σ̂_{tomorrow}
"""
from __future__ import annotations
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse existing pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))
from hartrace.features_v2 import build_features_v2, FAMILIES_V2  # type: ignore
from hartrace.estimation import fit_skewt_static  # type: ignore

logger = logging.getLogger(__name__)

FAMILY = 'HAR-RS-Q-WE-X'  # economisch beste model (G2)
REFIT_INTERVAL_DAYS = 20


@dataclass
class Forecast:
    asof_date: pd.Timestamp
    sigma_pred_daily: float           # σ̂ van log_RK forecast → daily vol
    log_rk_pred: float
    fit_age_days: int


class Forecaster:
    """Refit-cache + forecast pipeline."""
    
    def __init__(self, project_root: Path, family: str = FAMILY):
        self.project_root = project_root
        self.family = family
        self.fit_cache = project_root / 'state' / f'fit_{family}.pkl'
        self.fit_cache.parent.mkdir(parents=True, exist_ok=True)
    
    def load_features(self) -> pd.DataFrame:
        """Load and rebuild features from latest realized measures."""
        rv_path = self.project_root / 'data/processed/realized_measures_btc_eur_v2.parquet'
        fx_path = self.project_root / 'data/external/eurusd.parquet'
        fred_path = self.project_root / 'data/external/macro_fred.parquet'
        
        if not rv_path.exists():
            raise FileNotFoundError(
                f"Realized measures niet gevonden: {rv_path}. "
                f"Run scripts/B2_recompute_realized.py eerst."
            )
        
        feat = build_features_v2(
            rv_path, eurusd_path=fx_path, fred_path=fred_path, target='log_RK')
        return feat
    
    def _load_or_fit(self, feat: pd.DataFrame, force_refit: bool = False) -> dict:
        """Load cached fit or refit if stale."""
        family_cols = FAMILIES_V2[self.family]
        keep = [c for c in family_cols if c != 'const']
        
        cached = None
        if self.fit_cache.exists() and not force_refit:
            try:
                with open(self.fit_cache, 'rb') as f:
                    cached = pickle.load(f)
                age = (pd.Timestamp.utcnow().tz_localize(None) - cached['fit_date']).days
                if age <= REFIT_INTERVAL_DAYS:
                    logger.info(f"Using cached fit, age={age}d")
                    return cached
                else:
                    logger.info(f"Cached fit too old ({age}d), refitting...")
            except Exception as e:
                logger.warning(f"Could not load cached fit: {e}")
        
        # Refit on latest 1000 days
        df = feat[list(set(keep + ['target']))].dropna()
        sl = df.iloc[-1000:]
        X_tr = np.column_stack([np.ones(len(sl)), sl[keep].values]).astype(float)
        y_tr = sl['target'].values.astype(float)
        fit = fit_skewt_static(X_tr, y_tr, family_cols)
        fit['fit_date'] = pd.Timestamp.utcnow().tz_localize(None)
        fit['n_train'] = len(sl)
        with open(self.fit_cache, 'wb') as f:
            pickle.dump(fit, f)
        logger.info(f"Refit complete on n={len(sl)}: σ̂={fit['sigma']:.3f}, ν̂={fit['nu']:.1f}")
        return fit
    
    def forecast(self) -> Forecast:
        """Make tomorrow's σ̂ forecast using latest data."""
        feat = self.load_features()
        family_cols = FAMILIES_V2[self.family]
        keep = [c for c in family_cols if c != 'const']
        
        # Get features for most recent available date
        df = feat[list(set(keep + ['target']))].dropna()
        if len(df) == 0:
            raise RuntimeError("Geen feature data beschikbaar. Check data pipeline.")
        
        latest_date = df.index[-1]
        x_obs = df.iloc[-1][keep].values.astype(float)
        X_pred = np.concatenate([[1.0], x_obs])
        
        fit = self._load_or_fit(feat)
        log_rk_pred = float(X_pred @ fit['beta'])
        
        # σ̂ in daily-return terms = sqrt(exp(log_rk_pred))
        sigma_pred_daily = float(np.sqrt(np.exp(log_rk_pred)))
        
        fit_age = (pd.Timestamp.utcnow().tz_localize(None) - fit['fit_date']).days
        
        return Forecast(
            asof_date=latest_date,
            sigma_pred_daily=sigma_pred_daily,
            log_rk_pred=log_rk_pred,
            fit_age_days=fit_age,
        )
