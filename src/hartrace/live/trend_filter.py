"""Trend filter — pure MA50 price-trend signal voor BTC-EUR.

Strategie: weight = max_alloc als prijs > MA, anders 0.
Robust over sub-periodes (G5), cross-asset bevestigd (ETH-EUR), plateau in
MA-window keuze (40-80 dagen). Empirische winnaar op risk-adjusted basis.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    """Snapshot van de huidige trend-staat."""
    asof_date: pd.Timestamp
    current_price: float
    ma_value: float
    in_trend: bool             # True = price > MA, dus full position
    ma_window: int
    deviation_pct: float       # (price - ma) / ma; positief = boven trend
    n_history_days: int        # hoe veel dagen historie gebruikt


class TrendFilter:
    """Compute MA-based trend signal van daily closes."""
    
    def __init__(self, project_root: Path, ma_window: int = 50):
        self.project_root = Path(project_root)
        self.ma_window = ma_window
        self.candles_path = self.project_root / 'data/raw/candles_BTC-EUR_5m.parquet'
    
    def _load_daily_closes(self) -> pd.Series:
        """Daily close prices uit 5-min candles parquet."""
        if not self.candles_path.exists():
            raise FileNotFoundError(
                f"Candles file niet gevonden: {self.candles_path}. "
                f"Run scripts/btc_daily_data_update.py eerst."
            )
        candles = pd.read_parquet(self.candles_path)
        candles['date'] = pd.to_datetime(candles['ts'], unit='ms', utc=True
            ).dt.tz_localize(None).dt.normalize()
        # Daily close = laatste 5-min close van die dag
        daily = candles.groupby('date')['close'].last().sort_index()
        return daily
    
    def compute(self) -> TrendSignal:
        """Compute current trend signal."""
        daily = self._load_daily_closes()
        
        if len(daily) < self.ma_window + 1:
            raise RuntimeError(
                f"Need at least {self.ma_window + 1} dagen historie, "
                f"heb {len(daily)} dagen. Run data-update eerst."
            )
        
        latest_price = float(daily.iloc[-1])
        ma_value = float(daily.iloc[-self.ma_window:].mean())
        in_trend = latest_price > ma_value
        
        signal = TrendSignal(
            asof_date=daily.index[-1],
            current_price=latest_price,
            ma_value=ma_value,
            in_trend=in_trend,
            ma_window=self.ma_window,
            deviation_pct=(latest_price - ma_value) / ma_value,
            n_history_days=len(daily),
        )
        
        logger.info(f"Trend signal: price=€{latest_price:.2f}, "
                    f"MA{self.ma_window}=€{ma_value:.2f}, "
                    f"deviation={signal.deviation_pct*100:+.2f}%, "
                    f"in_trend={in_trend}")
        return signal
