"""Recompute daily realized measures from 5-min candles with v2 estimators.

Reads:  data/raw/candles_SOL-EUR_5m.parquet
Writes: data/processed/realized_measures_sol_eur_v2.parquet

The v2 set replaces v1 (15-min simple-RV) and adds:
    RK              Realized Kernel (Barndorff-Nielsen et al. 2008)
    realized_skew   intraday-return cubed skewness (Amaya et al. 2015)
    realized_kurt   intraday-return fourth moment / RV²
    signed_jump     RS_pos − RS_neg (Patton-Sheppard 2015)
    dow             0..6, day-of-week
    is_weekend      Sat/Sun flag
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from hartrace.realized_v2 import compute_v2_from_candles


PROJECT = Path(__file__).parent.parent
INPUT = PROJECT / 'data' / 'raw' / 'candles_SOL-EUR_5m.parquet'
OUT_DIR = PROJECT / 'data' / 'processed'
OUT_PARQUET = OUT_DIR / 'realized_measures_sol_eur_v2.parquet'
OUT_CSV = OUT_DIR / 'realized_measures_sol_eur_v2.csv'


def main():
    print("=" * 70)
    print("B2_recompute_realized.py — SOL-EUR realized measures v2")
    print("=" * 70)

    print(f"\n[1/3] Loading 5-min candles from {INPUT.name}")
    candles = pd.read_parquet(INPUT)
    print(f"  raw shape: {candles.shape}")
    candles['ts_dt'] = pd.to_datetime(candles['ts'], unit='ms', utc=True)
    print(f"  range    : {candles['ts_dt'].min()} → {candles['ts_dt'].max()}")
    print(f"  span     : {(candles['ts_dt'].max() - candles['ts_dt'].min()).days} days")

    print("\n[2/3] Computing v2 daily measures (RV, RK, BPV, RQ, RS±, jumps, ...)")
    t0 = time.time()
    df = compute_v2_from_candles(
        candles, ts_col='ts', close_col='close', volume_col='volume',
        ts_unit='ms', jump_alpha=0.999, min_bars=240,
    )
    print(f"  output    : {df.shape[0]} days × {df.shape[1]} columns ({time.time()-t0:.1f}s)")
    print(f"  date range: {df.index.min().date()} → {df.index.max().date()}")

    # Summary stats
    print("\n[3/3] Summary")
    print(f"\nRV vs RK comparison:")
    print(f"  mean    RV = {df['RV'].mean():.6f}")
    print(f"  mean    RK = {df['RK'].mean():.6f}")
    print(f"  RK/RV   ratio  = {df['RK'].mean()/df['RV'].mean():.3f} "
          f"(typically 0.7-0.85 if noise present)")
    print(f"\nAnnualised volatility (using RK):")
    ann = np.sqrt(df['RK'] * 365)
    print(f"  mean   = {ann.mean()*100:.1f}%")
    print(f"  median = {ann.median()*100:.1f}%")
    print(f"  range  = {ann.min()*100:.1f}% – {ann.max()*100:.1f}%")

    print(f"\nJump statistics (BNS α=0.001):")
    n_j = int(df['J_flag'].sum())
    print(f"  jump-days: {n_j} / {len(df)} ({100*n_j/len(df):.2f}%)")
    j_share = df['J'] / df['RV']
    j_share = j_share[df['J_flag'] == 1]
    if len(j_share) > 0:
        print(f"  J/RV on jump-days: mean={j_share.mean():.3f}, "
              f"max={j_share.max():.3f}")
    print(f"  mean signed_jump = {df['signed_jump'].mean():.6g} "
          f"(close to zero ⇒ symmetric)")

    print(f"\nRealized skew/kurt (intraday-return moments):")
    print(f"  realized_skew: mean={df['realized_skew'].mean():+.3f}, "
          f"median={df['realized_skew'].median():+.3f}")
    print(f"  realized_kurt: mean={df['realized_kurt'].mean():.2f}, "
          f"median={df['realized_kurt'].median():.2f}")

    print(f"\nSemi-variance asymmetry:")
    neg_share = df['RS_neg'] / (df['RS_neg'] + df['RS_pos'])
    print(f"  mean RS-/(RS+ + RS-) = {neg_share.mean():.4f}  (0.5 = symmetric)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(OUT_PARQUET, index=False)
    df.reset_index().to_csv(OUT_CSV, index=False)
    print(f"\nSaved:")
    print(f"  → {OUT_PARQUET.relative_to(PROJECT)}")
    print(f"  → {OUT_CSV.relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
