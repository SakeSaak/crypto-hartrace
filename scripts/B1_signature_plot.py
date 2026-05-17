"""Signature plot for BTC-EUR realized variance.

Question: at what sampling frequency does microstructure noise begin to
dominate the realized-variance estimate, and how much does noise-robust
estimation help?

Method
------
For each UTC day, take 1-minute log-prices and compute three estimators
at multiple sub-sampling frequencies (1, 2, 5, 10, 15, 30, 60 min):
    1. Simple RV       (Andersen-Bollerslev 1998)
    2. Two-Scale RV    (Zhang-Mykland-Aït-Sahalia 2005)
    3. Realized Kernel (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008)

Average across days and plot mean (RV / RV_simple_at_15min) vs frequency.
A flat plot means noise is negligible at all displayed frequencies;
a sloped plot means simple RV is upward-biased at fine frequencies, and
the noise-robust alternatives correct it.

Inputs
------
  data/raw/candles_BTC-EUR_1m.parquet  (may still be growing; we use
                                        whatever fraction is available)

Outputs
-------
  outputs/figures/B1_signature_plot.png
  outputs/tables/B1_signature_data.csv

Place in: crypto_hartrace/scripts/B1_signature_plot.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hartrace.rv_noise_robust import (
    simple_rv, two_scale_rv, realized_kernel,
)


PROJECT = Path(__file__).parent.parent
INPUT_1M = PROJECT / 'data' / 'raw' / 'candles_BTC-EUR_1m.parquet'
INPUT_5M = PROJECT / 'data' / 'raw' / 'candles_BTC-EUR_5m.parquet'
FIG_DIR = PROJECT / 'outputs' / 'figures'
TBL_DIR = PROJECT / 'outputs' / 'tables'

FREQS = [1, 2, 5, 10, 15, 30, 60]  # in minutes (for 1m data: 1=1min, 5=5min, ...)
MIN_BARS_PER_DAY = 1200  # at 1m, require ≥83% of 1440


def compute_one_day(log_prices: np.ndarray, freqs_min: list, base_min: int = 1):
    """For one day's log-prices on a base_min-minute grid, compute RV
    measures at multiple sub-sampling frequencies.

    Returns dict: { (estimator, freq_min): value }.
    """
    out = {}
    for f_min in freqs_min:
        step = max(1, f_min // base_min)
        sub = log_prices[::step]
        if len(sub) < 10:
            out[('rv', f_min)] = np.nan
            out[('tsrv', f_min)] = np.nan
            out[('rk', f_min)] = np.nan
            continue
        r = np.diff(sub)
        out[('rv', f_min)] = simple_rv(r)
        out[('tsrv', f_min)] = two_scale_rv(r, K=5) if len(r) >= 30 else np.nan
        out[('rk', f_min)] = realized_kernel(r) if len(r) >= 30 else np.nan
    return out


def main():
    print("=" * 64)
    print("B1_signature_plot.py — BTC-EUR signature plot")
    print("=" * 64)

    # Prefer 1-min data; fall back to 5-min if 1m fetch isn't done yet
    if INPUT_1M.exists():
        base_min = 1
        print(f"\nUsing 1-min input: {INPUT_1M.name}")
        df = pd.read_parquet(INPUT_1M)
    else:
        base_min = 5
        print(f"\n1-min not available, falling back to 5-min: {INPUT_5M.name}")
        df = pd.read_parquet(INPUT_5M)

    df = df.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df['date'] = df['ts_dt'].dt.floor('D')
    df['log_close'] = np.log(df['close'])
    n_total_bars = len(df)
    n_days = df['date'].nunique()
    print(f"  total bars: {n_total_bars:,}, days: {n_days}, "
          f"range: {df['ts_dt'].min().date()} → {df['ts_dt'].max().date()}")

    # Filter freqs to those that fit our base-frequency
    valid_freqs = [f for f in FREQS if f >= base_min]
    print(f"  base = {base_min}min, freqs to compute: {valid_freqs}")

    # Aggregate per day
    print(f"\n[1/2] Computing measures per day (base = {base_min}min)")
    rows = []
    t0 = time.time()
    n_kept = 0
    min_bars_day = MIN_BARS_PER_DAY if base_min == 1 else MIN_BARS_PER_DAY // 5
    for date, day_df in df.groupby('date'):
        if len(day_df) < min_bars_day:
            continue
        prices = day_df['log_close'].values
        measures = compute_one_day(prices, valid_freqs, base_min=base_min)
        row = {'date': date}
        for (est, f), v in measures.items():
            row[f'{est}_{f}m'] = v
        rows.append(row)
        n_kept += 1
        if n_kept % 200 == 0:
            print(f"  {n_kept} days processed ({time.time()-t0:.1f}s)")
    print(f"  total: {n_kept} days kept, {time.time()-t0:.1f}s")

    out_df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(TBL_DIR / 'B1_signature_per_day.csv', index=False)

    # Aggregate
    print("\n[2/2] Mean across days + signature plot")
    summary_rows = []
    for f in valid_freqs:
        for est in ('rv', 'tsrv', 'rk'):
            col = f'{est}_{f}m'
            if col in out_df.columns:
                vals = out_df[col].dropna()
                summary_rows.append({
                    'estimator': est, 'freq_min': f,
                    'mean': float(vals.mean()),
                    'median': float(vals.median()),
                    'std': float(vals.std()),
                    'n_days': len(vals),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TBL_DIR / 'B1_signature_summary.csv', index=False)
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(summary.to_string(index=False))

    # Plot
    pivot = summary.pivot_table(index='freq_min', columns='estimator',
                                  values='mean')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Left: absolute mean
    ax[0].plot(pivot.index, pivot['rv'], 'o-', color='#2b6cb0',
                label='Simple RV')
    if 'tsrv' in pivot.columns:
        ax[0].plot(pivot.index, pivot['tsrv'], 's-', color='#d97706',
                    label='Two-Scale RV')
    if 'rk' in pivot.columns:
        ax[0].plot(pivot.index, pivot['rk'], '^-', color='#7c3aed',
                    label='Realized Kernel')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Sampling frequency (minutes)')
    ax[0].set_ylabel('Mean RV per day')
    ax[0].set_title('Signature plot — mean RV vs sampling freq')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Right: relative to 15-min Simple RV
    if 15 in pivot.index:
        ref = float(pivot.loc[15, 'rv'])
        for est, marker, col in [('rv', 'o', '#2b6cb0'),
                                    ('tsrv', 's', '#d97706'),
                                    ('rk', '^', '#7c3aed')]:
            if est in pivot.columns:
                ax[1].plot(pivot.index, pivot[est] / ref,
                            marker=marker, linestyle='-', color=col,
                            label={'rv': 'Simple RV',
                                   'tsrv': 'Two-Scale RV',
                                   'rk': 'Realized Kernel'}[est])
        ax[1].axhline(1.0, color='k', linestyle='--', lw=0.7, alpha=0.5)
        ax[1].set_xscale('log')
        ax[1].set_xlabel('Sampling frequency (minutes)')
        ax[1].set_ylabel('Ratio to Simple-RV at 15-min')
        ax[1].set_title('Relative bias / noise vs 15-min baseline')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

    fig.suptitle(f'BTC-EUR signature plot — base = {base_min}min, '
                  f'{n_kept} days', y=1.00, fontsize=12, fontweight='bold')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_fig = FIG_DIR / 'B1_signature_plot.png'
    fig.savefig(out_fig, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  saved → {out_fig.relative_to(PROJECT)}")
    print(f"  table → {(TBL_DIR / 'B1_signature_summary.csv').relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
