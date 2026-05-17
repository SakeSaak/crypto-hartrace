"""Compute daily realized measures for BTC-EUR from 15-min candles.

Inputs
------
  /Users/sakesaakstra/data/bitvavo/candles_cache/candles_BTC-EUR_15m.parquet

Outputs
-------
  data/processed/realized_measures_btc_eur_15m.parquet
  data/processed/realized_measures_btc_eur_15m.csv  (peek)

Run with:
    python scripts/01_compute_realized_measures.py
"""

from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.realized import compute_from_candles


PROJECT_DIR = Path(__file__).parent.parent
CANDLES_PATH = Path(
    "/Users/sakesaakstra/data/bitvavo/candles_cache/candles_BTC-EUR_15m.parquet"
)
OUT_DIR = PROJECT_DIR / "data" / "processed"
OUT_PARQUET = OUT_DIR / "realized_measures_btc_eur_15m.parquet"
OUT_CSV = OUT_DIR / "realized_measures_btc_eur_15m.csv"

CROSSCHECK_PATH = Path("/Users/sakesaakstra/data/bitvavo/realized_daily.parquet")


def main():
    print("=" * 64)
    print("01_compute_realized_measures.py — BTC-EUR @ 15-min")
    print("=" * 64)

    print(f"\n[1/4] Loading candles from {CANDLES_PATH.name}")
    candles = pd.read_parquet(CANDLES_PATH)
    print(f"  Raw shape: {candles.shape}")
    print(f"  Columns:   {list(candles.columns)}")
    candles['ts_dt'] = pd.to_datetime(candles['ts'], unit='ms', utc=True)
    print(f"  Time range: {candles['ts_dt'].min()} → {candles['ts_dt'].max()}")
    print(f"  Span:       {(candles['ts_dt'].max() - candles['ts_dt'].min()).days} days")

    print("\n[2/4] Computing realized measures (RV, BPV, RQ, TQ, RS±, Z, C, J)")
    df = compute_from_candles(
        candles,
        ts_col='ts', close_col='close', ts_unit='ms',
        jump_alpha=0.999, min_bars=80,
    )
    print(f"  Output shape: {df.shape}")
    print(f"  Date range:   {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Days kept:    {len(df)} / {(df.index.max() - df.index.min()).days + 1}")

    print("\n[3/4] Summary statistics")
    # Annualised quantities (×365 since 24/7 crypto market)
    df_disp = df.copy()
    df_disp['vol_ann'] = np.sqrt(df['RV'] * 365.0)
    cols = ['RV', 'BPV', 'RQ', 'RS_neg', 'RS_pos', 'C', 'J', 'Z', 'J_flag',
            'r_d', 'vol_ann', 'n_bars']
    with pd.option_context('display.float_format', '{:.6f}'.format,
                            'display.width', 140):
        print(df_disp[cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

    n_jumps = int(df['J_flag'].sum())
    print(f"\n  Jump-days (BNS @ α=0.001): {n_jumps} / {len(df)} "
          f"({100 * n_jumps / len(df):.1f}%)")
    print(f"  Mean RS_neg / (RS_neg + RS_pos): "
          f"{(df['RS_neg'] / (df['RS_neg'] + df['RS_pos'])).mean():.4f} "
          f"(downside fraction; symmetric markt → 0.50)")
    print(f"  Mean J / RV (jump share):       "
          f"{(df['J'] / df['RV']).replace([np.inf, -np.inf], np.nan).mean():.4f}")
    rho_logRV = df['RV'].apply(np.log).autocorr()
    print(f"  ACF(1) of log RV:               {rho_logRV:.4f} (long-memory check)")

    # Cross-check against existing realized_daily.parquet
    if CROSSCHECK_PATH.exists():
        print("\n[3b/4] Cross-validation against existing realized_daily.parquet")
        rd = pd.read_parquet(CROSSCHECK_PATH)
        rd = rd[rd['market'] == 'BTC-EUR'].copy()
        rd['date'] = pd.to_datetime(rd['date_utc']).dt.tz_localize(None).dt.normalize()
        rd = rd.set_index('date')[['rv_5m', 'rv_1h']]
        merged = df[['RV']].copy()
        merged.index = merged.index.tz_localize(None)
        merged = merged.join(rd, how='inner')
        if len(merged) > 0:
            corr_5m = merged['RV'].corr(merged['rv_5m'])
            corr_1h = merged['RV'].corr(merged['rv_1h'])
            print(f"  Overlap: {len(merged)} days")
            print(f"  corr(RV_15m, RV_5m existing): {corr_5m:.4f}")
            print(f"  corr(RV_15m, RV_1h existing): {corr_1h:.4f}")
            print(f"  mean RV_15m / RV_5m ratio: "
                  f"{(merged['RV'] / merged['rv_5m']).replace([np.inf,-np.inf],np.nan).mean():.4f}")

    print(f"\n[4/4] Writing output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(OUT_PARQUET, index=False)
    df.reset_index().to_csv(OUT_CSV, index=False)
    print(f"  → {OUT_PARQUET}")
    print(f"  → {OUT_CSV}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
