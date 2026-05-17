"""Bitvavo REST historical candles fetcher.

Polite, rate-limited, resumable fetcher for OHLCV candles via Bitvavo's
public REST API (`/v2/{market}/candles`).

Design choices
--------------
- Walks backwards from `end_dt` (default: now) until either `start_dt`
  is reached or Bitvavo returns no more data (market launch reached).
- Each REST call retrieves up to 1 440 candles.
- Tracks `Bitvavo-Ratelimit-Remaining` header and throttles politely.
- Exponential backoff on 429 and 5xx responses.
- Resumable: detects existing `candles_{market}_{interval}.parquet`
  and continues from its earliest timestamp.
- Atomic incremental saves every 50 calls so a crash mid-fetch doesn't
  lose progress.

Place in: crypto_hartrace/src/hartrace/bitvavo_rest.py

Usage:
    from hartrace.bitvavo_rest import BitvavoCandleFetcher
    f = BitvavoCandleFetcher('BTC-EUR', '5m', 'data/raw')
    df = f.fetch()    # fetches all available history
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.bitvavo.com/v2"

INTERVAL_MS: dict[str, int] = {
    '1m':  60_000,
    '5m':  5 * 60_000,
    '15m': 15 * 60_000,
    '30m': 30 * 60_000,
    '1h':  60 * 60_000,
    '2h':  2 * 60 * 60_000,
    '4h':  4 * 60 * 60_000,
    '6h':  6 * 60 * 60_000,
    '8h':  8 * 60 * 60_000,
    '12h': 12 * 60 * 60_000,
    '1d':  24 * 60 * 60_000,
}

MAX_LIMIT = 1440


def _ms_now() -> int:
    return int(time.time() * 1000)


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


class BitvavoCandleFetcher:
    """Backwards-walking, resumable Bitvavo candles fetcher.

    Parameters
    ----------
    market : str
        Market symbol, e.g. 'BTC-EUR'.
    interval : str
        One of '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h',
        '8h', '12h', '1d'.
    out_dir : path-like
        Directory in which `candles_{market}_{interval}.parquet` is written.
    throttle_s : float
        Minimum seconds between successive REST calls. The fetcher
        additionally honours the rate-limit header.
    """

    def __init__(
        self,
        market: str,
        interval: str,
        out_dir: str | Path,
        throttle_s: float = 0.15,
    ):
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unknown interval {interval!r}; "
                             f"expected one of {sorted(INTERVAL_MS)}")
        self.market = market
        self.interval = interval
        self.interval_ms = INTERVAL_MS[interval]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"candles_{market}_{interval}.parquet"
        self.throttle_s = throttle_s

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'crypto_hartrace/0.1 (academic research)',
            'Accept': 'application/json',
        })

    # ------------------------------------------------------------------
    # Private REST plumbing
    # ------------------------------------------------------------------
    def _request(self, params: dict) -> list:
        url = f"{BASE_URL}/{self.market}/candles"
        for attempt in range(5):
            try:
                r = self.session.get(url, params=params, timeout=20)
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(f"   network error ({exc!r}); sleeping {wait}s ...")
                time.sleep(wait)
                continue

            if r.status_code == 200:
                # Polite throttle: extra delay if rate-limit budget low
                try:
                    rem = int(r.headers.get('Bitvavo-Ratelimit-Remaining', '900'))
                except ValueError:
                    rem = 900
                time.sleep(2.0 if rem < 100 else self.throttle_s)
                return r.json()

            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"   429 rate-limited; sleeping {wait}s ...")
                time.sleep(wait)
                continue

            if r.status_code >= 500:
                wait = 2 ** attempt
                print(f"   {r.status_code} server error; sleeping {wait}s ...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Bitvavo {r.status_code}: {r.text[:200]}")

        raise RuntimeError(f"Gave up after 5 retries on {url} {params}")

    def _fetch_batch(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """One REST call. Returns DataFrame with columns ts, open, high, low, close, volume."""
        params = {
            'interval': self.interval,
            'limit': MAX_LIMIT,
            'start': int(start_ms),
            'end': int(end_ms),
        }
        raw = self._request(params)
        if not raw:
            return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = df['ts'].astype('int64')
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = df[col].astype(float)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_existing(self) -> Optional[pd.DataFrame]:
        if self.out_path.exists():
            return pd.read_parquet(self.out_path)
        return None

    def _save_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.out_path.with_suffix('.parquet.tmp')
        df.to_parquet(tmp, index=False)
        tmp.replace(self.out_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch(
        self,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        max_calls: int = 5000,
    ) -> pd.DataFrame:
        """Fetch backwards from end_dt to start_dt, resuming existing parquet.

        Parameters
        ----------
        start_dt : datetime, optional
            Earliest desired timestamp (UTC). If None, fetches until empty
            response (Bitvavo retention limit / market launch).
        end_dt : datetime, optional
            Latest desired timestamp (UTC). Default: now.
        max_calls : int
            Safety bound on number of REST calls.

        Returns
        -------
        pd.DataFrame
            The full local dataset (existing + newly fetched), sorted, deduplicated.
        """
        end_ms = int(end_dt.timestamp() * 1000) if end_dt else _ms_now()
        start_ms = int(start_dt.timestamp() * 1000) if start_dt else 0

        existing = self._load_existing()
        if existing is not None and len(existing) > 0:
            earliest = int(existing['ts'].min())
            latest = int(existing['ts'].max())
            print(f"   resumable: existing {len(existing):,} bars, "
                  f"{_ms_to_utc(earliest).date()} → {_ms_to_utc(latest).date()}")
            # We fetch backwards from earliest existing
            end_ms = min(end_ms, earliest - 1)

        new_frames: list[pd.DataFrame] = []
        cursor = end_ms
        call_count = 0
        empty_streak = 0

        while cursor > start_ms and call_count < max_calls and empty_streak < 3:
            window_start = max(start_ms, cursor - MAX_LIMIT * self.interval_ms)
            batch = self._fetch_batch(window_start, cursor)
            call_count += 1

            if batch.empty:
                empty_streak += 1
                print(f"   call {call_count:4d}: empty at {_ms_to_utc(cursor).date()} "
                      f"(empty_streak={empty_streak})")
                # Step back one window and try again
                cursor = window_start - 1
                continue

            empty_streak = 0
            new_frames.append(batch)
            tmin = int(batch['ts'].min())
            tmax = int(batch['ts'].max())
            print(f"   call {call_count:4d}: {len(batch):4d} bars, "
                  f"{_ms_to_utc(tmin).date()} → {_ms_to_utc(tmax).date()}")

            new_cursor = tmin - 1
            if new_cursor >= cursor:   # safety: not making progress
                print("   not advancing — stopping")
                break
            cursor = new_cursor

            # Incremental save every 50 calls
            if call_count % 50 == 0:
                self._merge_and_save(new_frames, existing)

        # Final save
        full = self._merge_and_save(new_frames, existing)
        if full is None or len(full) == 0:
            print("   no data fetched and no existing data.")
            return pd.DataFrame()

        print(f"   final: {len(full):,} bars, "
              f"{_ms_to_utc(int(full['ts'].min())).date()} → "
              f"{_ms_to_utc(int(full['ts'].max())).date()} "
              f"({call_count} calls)")
        return full

    def _merge_and_save(
        self,
        new_frames: list[pd.DataFrame],
        existing: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        parts = []
        if existing is not None and len(existing) > 0:
            parts.append(existing)
        if new_frames:
            parts.append(pd.concat(new_frames, ignore_index=True))
        if not parts:
            return None
        full = (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset='ts')
            .sort_values('ts')
            .reset_index(drop=True)
        )
        self._save_atomic(full)
        return full


# ----------------------------------------------------------------------
# Convenience top-level API
# ----------------------------------------------------------------------
def fetch_all_intervals(
    market: str,
    intervals: list[str],
    out_dir: str | Path,
    throttle_s: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Fetch a set of intervals sequentially for one market. Returns dict by interval."""
    out: dict[str, pd.DataFrame] = {}
    for ivl in intervals:
        print(f"\n=== {market} {ivl} ===")
        out[ivl] = BitvavoCandleFetcher(market, ivl, out_dir, throttle_s).fetch()
    return out
