"""Fetch external macro/FX data for HAR-MIDAS regressors.

Pulls three datasets at daily frequency:
  - data/external/eurusd.parquet      (yfinance)
  - data/external/btc_dominance.parquet  (CoinGecko)
  - data/external/macro_fred.parquet  (FRED, needs FRED_API_KEY)

Requires:
  pip install yfinance fredapi
  Register a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
  export FRED_API_KEY="your-key-here"

Run:
  python scripts/A2_fetch_external_data.py

Place in: crypto_hartrace/scripts/A2_fetch_external_data.py
"""

from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hartrace.external_data import (
    fetch_eurusd, fetch_btc_dominance, fetch_fred_series,
    DEFAULT_FRED_SERIES,
)

PROJECT = Path(__file__).parent.parent
OUT_DIR = PROJECT / 'data' / 'external'
START = '2018-01-01'  # voor de zekerheid een jaar vóór Bitvavo's BTC-EUR launch


def main():
    print("=" * 70)
    print("A2_fetch_external_data.py — EUR/USD + BTC-dominance + FRED macro")
    print("=" * 70)
    print(f"Output dir: {OUT_DIR.relative_to(PROJECT)}")
    print(f"Start    : {START}")

    # 1) EUR/USD via yfinance
    print("\n[1/3] EUR/USD FX")
    try:
        fetch_eurusd(OUT_DIR / 'eurusd.parquet', start=START)
    except Exception as exc:
        print(f"   FAILED: {exc!r}")

    # 2) BTC dominance via CoinGecko
    print("\n[2/3] BTC dominance")
    try:
        fetch_btc_dominance(OUT_DIR / 'btc_dominance.parquet', start=START)
    except Exception as exc:
        print(f"   FAILED: {exc!r}")
        print("   (CoinGecko free tier kan rate-limited zijn; "
              "probeer later opnieuw of gebruik scrape-fallback)")

    # 3) FRED macro
    print("\n[3/3] FRED macro factors")
    if not os.environ.get('FRED_API_KEY'):
        print("   SKIP: set FRED_API_KEY env var first")
        print("   (free signup: https://fred.stlouisfed.org/docs/api/api_key.html)")
    else:
        try:
            fetch_fred_series(
                DEFAULT_FRED_SERIES,
                OUT_DIR / 'macro_fred.parquet',
                start=START,
            )
        except Exception as exc:
            print(f"   FAILED: {exc!r}")

    print("\nOn-disk summary:")
    for p in sorted(OUT_DIR.glob('*.parquet')):
        sz = p.stat().st_size / 1024
        print(f"  {p.name:<30} {sz:8.1f} KB")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
