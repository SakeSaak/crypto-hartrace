"""ATR-stop + Trend MA50 filter — H3-robustness winner.

Stateful strategy:
  - Entry: NOT in_position AND price > MA50 → enter, set stop = price - mult × ATR
  - In position: update trailing stop (kan alleen omhoog), exit als price < stop
  - Geen explicit MA50-exit; pure stop-driven exit

State persistence required: in_position bool + current_stop level.
Live: gebruikt OHLC voor proper True Range (max van H-L, |H-prevC|, |L-prevC|).
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AtrTrendSignal:
    asof_date: pd.Timestamp
    current_price: float
    ma_value: float
    atr_value: float
    in_position_prior: bool       # state binnenkomend uit vorige run
    current_stop_prior: float     # stop-level binnenkomend
    new_stop: float                # bijgewerkt stop-level (na ratcheting)
    in_position_after: bool       # state na deze beslissing
    target_w: float                # 0.90 of 0.00
    action: str                    # 'enter' / 'exit' / 'hold' / 'wait' / 'update_stop'
    ma_window: int
    atr_window: int
    atr_multiplier: float
    deviation_pct: float           # (price - ma) / ma
    stop_distance_pct: float       # (price - stop) / price wanneer in_position


class AtrTrendFilter:
    """ATR-trailing stop op Trend MA50 entry. Stateful."""

    def __init__(self, project_root: Path, ma_window: int = 50,
                 atr_window: int = 14, atr_multiplier: float = 3.0):
        self.project_root = Path(project_root)
        self.ma_window = ma_window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.candles_path = self.project_root / 'data/raw/candles_BTC-EUR_5m.parquet'

    def _load_daily_ohlc(self) -> pd.DataFrame:
        """Daily OHLC uit 5-min candles (groupby per dag)."""
        if not self.candles_path.exists():
            raise FileNotFoundError(
                f"Candles file niet gevonden: {self.candles_path}. "
                f"Run scripts/btc_daily_data_update.py eerst."
            )
        df = pd.read_parquet(self.candles_path)
        df['date'] = pd.to_datetime(df['ts'], unit='ms', utc=True
            ).dt.tz_localize(None).dt.normalize()
        daily = df.groupby('date').agg(
            open=('open', 'first'), high=('high', 'max'),
            low=('low', 'min'), close=('close', 'last')
        ).sort_index()
        return daily

    @staticmethod
    def _true_range(daily: pd.DataFrame) -> pd.Series:
        """True Range = max(H-L, |H-prevC|, |L-prevC|)."""
        prev_close = daily['close'].shift(1)
        tr = pd.concat([
            daily['high'] - daily['low'],
            (daily['high'] - prev_close).abs(),
            (daily['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr

    def compute(self, in_position_prior: bool, current_stop_prior: float) -> AtrTrendSignal:
        """Stateful signal. Neem huidige state, return signal + nieuwe state."""
        daily = self._load_daily_ohlc()
        min_needed = max(self.ma_window, self.atr_window) + 1
        if len(daily) < min_needed:
            raise RuntimeError(
                f"Need at least {min_needed} dagen, heb {len(daily)}. "
                f"Run data-update eerst."
            )

        price = float(daily['close'].iloc[-1])
        ma = float(daily['close'].iloc[-self.ma_window:].mean())
        tr = self._true_range(daily)
        atr = float(tr.iloc[-self.atr_window:].mean())

        # Stop-candidate o.b.v. huidige prijs en ATR
        candidate_stop = price - self.atr_multiplier * atr

        # Beslissingslogica
        if not in_position_prior:
            # Kandidaat voor entry
            if price > ma:
                in_position_after = True
                new_stop = candidate_stop
                target_w = 0.90
                action = 'enter'
            else:
                in_position_after = False
                new_stop = -math.inf
                target_w = 0.0
                action = 'wait'
        else:
            # In positie — check stop
            new_stop = max(current_stop_prior, candidate_stop)  # alleen omhoog
            if price < current_stop_prior:  # stop is GERAAKT (price onder oude stop)
                in_position_after = False
                new_stop = -math.inf
                target_w = 0.0
                action = 'exit'
            else:
                in_position_after = True
                target_w = 0.90
                if new_stop > current_stop_prior + 1e-9:
                    action = 'update_stop'
                else:
                    action = 'hold'

        signal = AtrTrendSignal(
            asof_date=daily.index[-1],
            current_price=price,
            ma_value=ma,
            atr_value=atr,
            in_position_prior=in_position_prior,
            current_stop_prior=current_stop_prior,
            new_stop=new_stop,
            in_position_after=in_position_after,
            target_w=target_w,
            action=action,
            ma_window=self.ma_window,
            atr_window=self.atr_window,
            atr_multiplier=self.atr_multiplier,
            deviation_pct=(price - ma) / ma,
            stop_distance_pct=(price - current_stop_prior) / price 
                if in_position_prior and current_stop_prior > 0 else float('nan'),
        )

        logger.info(
            f"ATR-stop signal: price=€{price:.2f}, MA{self.ma_window}=€{ma:.2f} "
            f"({signal.deviation_pct*100:+.2f}%), ATR={atr:.2f}, "
            f"prior_pos={in_position_prior}, prior_stop=€{current_stop_prior:.2f}, "
            f"new_stop=€{new_stop:.2f}, action={action}, target_w={target_w:.2f}"
        )
        return signal
