"""A1b — Fetch SOL-EUR candles voor cross-asset robustness (Optie B-light)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from hartrace.bitvavo_rest import BitvavoCandleFetcher

RAW_DIR = Path('/Users/sakesaakstra/Desktop/crypto_hartrace/data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Alleen 5m frequency (nodig voor realized variance) en 1d (voor sanity)
for interval in ['1d', '5m']:
    print(f"\n=== SOL-EUR {interval} ===")
    f = BitvavoCandleFetcher('SOL-EUR', interval, RAW_DIR, throttle_s=0.15)
    try:
        f.fetch(max_calls=4000)
    except Exception as e:
        print(f"FAILED: {e!r}")
print("\nDone.")
