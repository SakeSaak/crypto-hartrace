"""Fetch historical OHLCV candles for BTC-EUR (and ETH-EUR) from Bitvavo.

Fetches the full available history at multiple sampling intervals:

    1m   → finest resolution, needed for signature-plot diagnostics + 
            for testing whether noise-robust estimators can use 1m
    5m   → ABDL (2001) canonical RV sampling frequency
    15m  → conservative, lower microstructure-noise pressure
    1h   → robustness check / regime detection
    1d   → daily anchor for long-history validation

Run with:
    python scripts/A1_fetch_bitvavo_candles.py

Output:
    data/raw/candles_BTC-EUR_{interval}.parquet  (one per interval)
    data/raw/candles_ETH-EUR_{interval}.parquet  (idem, for cross-asset robustness)

The script is resumable — re-running picks up where it left off based on
the earliest timestamp already on disk. Politely rate-limited per
Bitvavo's `Bitvavo-Ratelimit-Remaining` header.

Place in: crypto_hartrace/scripts/A1_fetch_bitvavo_candles.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hartrace.bitvavo_rest import BitvavoCandleFetcher

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / 'data' / 'raw'

# Primary asset(s). Add ETH-EUR for cross-asset validation if desired.
MARKETS = ['BTC-EUR', 'ETH-EUR']

# Sampling intervals to fetch — descending from coarsest to finest is
# faster (fewer calls, more candles per call) and lets us check on
# coarser data while finer fetches continue.
INTERVALS = ['1d', '1h', '15m', '5m', '1m']

# Politeness throttle in seconds between successive calls.
# Bitvavo's typical retail-app weight-budget is ~1000/minute; one
# candles-call costs 1 weight. At 0.15s between calls (≈400/min) we
# leave ample headroom for the live order-collector and any other
# concurrent activity.
THROTTLE_S = 0.15

# Per-market cap on number of REST calls per interval. Generous default;
# the loop exits early once empty responses (= history limit) are reached.
MAX_CALLS = 4000


def main():
    print("=" * 72)
    print("A1_fetch_bitvavo_candles.py — full Bitvavo historical candles")
    print("=" * 72)
    print(f"Markets   : {MARKETS}")
    print(f"Intervals : {INTERVALS}")
    print(f"Output    : {RAW_DIR.relative_to(PROJECT)}")
    print(f"Throttle  : {THROTTLE_S}s between calls")

    t0 = time.time()
    for market in MARKETS:
        for interval in INTERVALS:
            print(f"\n=== {market} {interval} ===")
            f = BitvavoCandleFetcher(market, interval, RAW_DIR,
                                       throttle_s=THROTTLE_S)
            try:
                f.fetch(max_calls=MAX_CALLS)
            except Exception as exc:
                print(f"   FAILED: {exc!r}")
                print("   (continuing to next interval)")
                continue
    dt = time.time() - t0
    print(f"\nTotal elapsed: {dt:.1f}s ({dt/60:.1f} min)")

    print("\nOn-disk summary:")
    for p in sorted(RAW_DIR.glob('candles_*.parquet')):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<40} {size_mb:6.1f} MB")


if __name__ == '__main__':
    main()
