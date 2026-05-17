"""Daily data-update job — runs vóór de trader.

1. Fetch nieuwe Bitvavo 5-min candles sinds laatste update (public endpoint)
2. Append aan bestaande candles parquet
3. Run B2 om realized_measures parquet te updaten
4. Run A2 om EUR/USD + FRED macro te updaten

Schedule via launchd om 22:00 UTC (5 min vóór trader-run om 22:05 UTC).
"""
from __future__ import annotations
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd

PROJECT = Path(__file__).parent.parent
PYTHON = '/opt/anaconda3/bin/python'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(PROJECT / 'logs' / f'data_update_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def update_candles_incremental() -> int:
    """Fetch nieuwe Bitvavo 5m candles sinds laatste in onze parquet."""
    from hartrace.bitvavo_rest import BitvavoRESTFetcher
    
    raw_path = PROJECT / 'data/raw/candles_BTC-EUR_5m.parquet'
    if not raw_path.exists():
        log.error(f"Initial candles file niet gevonden: {raw_path}. "
                   f"Run scripts/A1_fetch_bitvavo_candles.py eerst.")
        return 0
    
    existing = pd.read_parquet(raw_path)
    last_ts_ms = int(existing['ts'].max())
    last_dt = datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)
    
    if (now_dt - last_dt) < timedelta(minutes=10):
        log.info(f"Candles up-to-date ({last_dt.isoformat()}). Skipping fetch.")
        return 0
    
    log.info(f"Fetching candles vanaf {last_dt.isoformat()} tot nu...")
    fetcher = BitvavoRESTFetcher()
    
    new = fetcher.fetch_candles(
        market='BTC-EUR', interval='5m',
        start_ts_ms=last_ts_ms + 1, end_ts_ms=int(now_dt.timestamp() * 1000),
        rate_limit_pause=0.05,
    )
    
    if new is None or len(new) == 0:
        log.info("Geen nieuwe candles ontvangen.")
        return 0
    
    log.info(f"  ontvangen: {len(new)} nieuwe candles")
    # Concat + dedup op ts
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset='ts', keep='last').sort_values('ts').reset_index(drop=True)
    
    n_added = len(combined) - len(existing)
    if n_added > 0:
        combined.to_parquet(raw_path, index=False)
        log.info(f"  toegevoegd: {n_added} rows → totaal {len(combined)} candles")
    return n_added


def run_b2_recompute():
    """Run B2 om realized measures te recompute."""
    log.info("Running B2_recompute_realized.py...")
    r = subprocess.run(
        [PYTHON, '-u', str(PROJECT / 'scripts/B2_recompute_realized.py')],
        capture_output=True, text=True, cwd=str(PROJECT)
    )
    if r.returncode != 0:
        log.error(f"B2 failed (rc={r.returncode}):\n{r.stderr[:500]}")
        return False
    # Print last 5 lines of output
    out_lines = r.stdout.strip().split('\n')
    for line in out_lines[-5:]:
        log.info(f"  B2: {line}")
    return True


def run_a2_external_update():
    """Update EUR/USD + FRED macro. Tolerant voor falen (geen blocker)."""
    log.info("Running A2_fetch_external_data.py...")
    r = subprocess.run(
        [PYTHON, '-u', str(PROJECT / 'scripts/A2_fetch_external_data.py')],
        capture_output=True, text=True, cwd=str(PROJECT), timeout=120,
    )
    if r.returncode != 0:
        log.warning(f"A2 failed (rc={r.returncode}). Doorgaan met oude macro data.")
        return False
    log.info("  A2 OK")
    return True


def main():
    log.info("=" * 60)
    log.info("Daily data-update job")
    log.info("=" * 60)
    t0 = time.time()
    
    try:
        n_new = update_candles_incremental()
    except Exception as e:
        log.exception(f"Candle fetch failed: {e}")
        return 1
    
    if n_new > 0:
        if not run_b2_recompute():
            return 2
    else:
        log.info("Geen nieuwe candles → B2 skip")
    
    # External data (best effort)
    run_a2_external_update()
    
    elapsed = time.time() - t0
    log.info(f"Daily data-update klaar in {elapsed:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
