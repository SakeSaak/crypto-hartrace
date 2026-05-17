"""Economic value test — vol-managed strategy on HAR-family forecasts.

OOS-periode: 2022-09 t/m 2026-05, n=1312 dagen (van E1 walk-forward).

Strategieën (alle euro-denominated, daily rebalancing):
  - Buy-and-hold        : w_t = 1 (passief)
  - Vol-managed HAR-RS-DOW : w_t = target_σ / σ̂_{t+1|t}^DOW
  - Vol-managed HAR-WE     : w_t = target_σ / σ̂_{t+1|t}^WE
  - Oracle             : w_t = target_σ / σ_realized_{t+1}  (perfect foresight)

Target annual vol = 50% (matcht BTC's lange-termijn realized vol).
Position cap = 3x leverage (realistisch institutional limit).
Transactiekosten: 15 bps per zijde (Bitvavo retail spread + fees).

Metrieken: total return, annualized vol, Sharpe, Sortino, max drawdown,
Calmar, turnover. Gross én net van TC.

Plot: equity curves, sigma forecasts vs realized, position weights.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
TBL = PROJECT / 'outputs' / 'tables'
FIG = PROJECT / 'outputs' / 'figures'
SUM = PROJECT / 'outputs' / 'G1_economic_value_summary.md'

TARGET_VOL_ANNUAL = 0.50      # 50% annual vol target
TARGET_VOL_DAILY = TARGET_VOL_ANNUAL / np.sqrt(365)
MAX_LEVERAGE = 3.0            # cap on position weight
TC_RATE = 0.0015              # 15 bps per side, Bitvavo retail


def load_forecasts(fam: str) -> pd.DataFrame:
    """Load E1 walk-forward parquet, extract mu_pred and r_actual."""
    df = pd.read_parquet(TBL / f'E1_walkforward_per_day_{fam}.parquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # daily σ̂ = sqrt(exp(mu_pred_logRK))
    df['sigma_daily_pred'] = np.sqrt(np.exp(df['mu_pred']))
    return df[['mu_pred', 'sigma_daily_pred', 'y_actual', 'r_actual']]


def load_realized_for_oracle(dates) -> pd.Series:
    """Realized daily σ (sqrt RK) aligned to date index — oracle benchmark."""
    rv = pd.read_parquet(PROJECT / 'data/processed/realized_measures_btc_eur_v2.parquet')
    rv['date'] = pd.to_datetime(rv['date'])
    rv = rv.set_index('date').sort_index()
    sigma_realized = np.sqrt(rv['RK'])
    # σ̂ for "tomorrow's return" = today's realized vol shifted forward
    sigma_realized_shifted = sigma_realized.shift(-1)  # σ on day t+1
    return sigma_realized_shifted.reindex(dates)


def compute_weights(sigma_pred: pd.Series) -> pd.Series:
    """Position weight = target_daily / σ̂, capped at MAX_LEVERAGE."""
    w = TARGET_VOL_DAILY / sigma_pred
    return w.clip(upper=MAX_LEVERAGE)


def metrics(returns: pd.Series, label: str) -> dict:
    """Annualized Sharpe, Sortino, MDD, total return etc."""
    r = returns.dropna()
    n = len(r)
    mean_d = float(r.mean())
    std_d = float(r.std(ddof=1))
    sharpe = np.sqrt(365) * mean_d / std_d if std_d > 0 else np.nan
    downside = r[r < 0]
    sortino = (np.sqrt(365) * mean_d / downside.std(ddof=1)
                if len(downside) > 1 else np.nan)
    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    annualized_return = float((1 + total_return) ** (365 / n) - 1)
    annualized_vol = float(std_d * np.sqrt(365))
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    mdd = float(dd.min())
    calmar = annualized_return / abs(mdd) if mdd < 0 else np.nan
    pct_positive = float((r > 0).mean())
    return {
        'label': label, 'n': n,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': mdd, 'calmar': calmar,
        'pct_positive': pct_positive,
    }


def build_strategy(forecasts: pd.DataFrame, label: str,
                    sigma_pred_col: str = 'sigma_daily_pred',
                    apply_tc: bool = True) -> pd.DataFrame:
    """Build vol-managed strategy returns from σ̂ forecasts."""
    df = forecasts.copy()
    df['weight'] = compute_weights(df[sigma_pred_col])
    df['weight_lag'] = df['weight'].shift(1).fillna(1.0)
    df['turnover'] = (df['weight'] - df['weight_lag']).abs()
    df['r_gross'] = df['weight'] * df['r_actual']
    df['tc'] = TC_RATE * df['turnover']
    df['r_net'] = df['r_gross'] - df['tc']
    df['strategy'] = label
    return df


def main():
    print("=" * 72)
    print("G1 — Economic value test: vol-managed vs buy-and-hold")
    print("=" * 72)
    print(f"Target annual vol = {TARGET_VOL_ANNUAL*100:.0f}%")
    print(f"Max leverage     = {MAX_LEVERAGE}×")
    print(f"Transaction cost = {TC_RATE*100:.2f}% per side")
    print()

    # Load three model forecasts
    fams = ['HAR-RS-DOW', 'HAR-WE', 'HAR-RS-Q-WE-X']
    forecasts = {f: load_forecasts(f) for f in fams}

    # Common date index
    dates = forecasts[fams[0]].index
    print(f"OOS-periode: {dates.min().date()} → {dates.max().date()} (n={len(dates)})")
    print()

    # Build strategies
    strats = {}
    for f in fams:
        df = build_strategy(forecasts[f], f)
        strats[f] = df

    # Oracle (perfect foresight)
    sigma_oracle = load_realized_for_oracle(dates).reindex(dates)
    df_oracle = forecasts[fams[0]][['r_actual']].copy()
    df_oracle['weight'] = compute_weights(sigma_oracle)
    df_oracle['weight_lag'] = df_oracle['weight'].shift(1).fillna(1.0)
    df_oracle['turnover'] = (df_oracle['weight'] - df_oracle['weight_lag']).abs()
    df_oracle['r_gross'] = df_oracle['weight'] * df_oracle['r_actual']
    df_oracle['tc'] = TC_RATE * df_oracle['turnover']
    df_oracle['r_net'] = df_oracle['r_gross'] - df_oracle['tc']

    # Buy-and-hold
    df_bh = forecasts[fams[0]][['r_actual']].copy()
    df_bh['weight'] = 1.0
    df_bh['turnover'] = 0.0
    df_bh['r_gross'] = df_bh['r_actual']
    df_bh['r_net'] = df_bh['r_gross']

    # Metrics
    print("-" * 80)
    print(f"{'Strategy':<26}{'TotRet':>9}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>9}{'Calmar':>8}")
    print("-" * 80)

    all_metrics = []
    rows_for_tc = []

    # Buy-and-hold
    m = metrics(df_bh['r_gross'], 'Buy-and-hold')
    all_metrics.append(m)
    print(f"{'Buy-and-hold':<26}"
          f"{m['total_return']*100:>8.1f}%"
          f"{m['annualized_return']*100:>8.1f}%"
          f"{m['annualized_vol']*100:>8.1f}%"
          f"{m['sharpe']:>8.2f}"
          f"{m['sortino']:>9.2f}"
          f"{m['max_drawdown']*100:>8.1f}%"
          f"{m['calmar']:>8.2f}")

    # Vol-managed (gross and net)
    for f in fams:
        df = strats[f]
        m_g = metrics(df['r_gross'], f'{f} (gross)')
        m_n = metrics(df['r_net'], f'{f} (net 15bps)')
        all_metrics.append(m_g)
        all_metrics.append(m_n)
        for m in (m_g, m_n):
            print(f"{m['label']:<26}"
                  f"{m['total_return']*100:>8.1f}%"
                  f"{m['annualized_return']*100:>8.1f}%"
                  f"{m['annualized_vol']*100:>8.1f}%"
                  f"{m['sharpe']:>8.2f}"
                  f"{m['sortino']:>9.2f}"
                  f"{m['max_drawdown']*100:>8.1f}%"
                  f"{m['calmar']:>8.2f}")

    # Oracle
    m = metrics(df_oracle['r_gross'], 'Oracle vol-managed (gross)')
    all_metrics.append(m)
    m_n = metrics(df_oracle['r_net'], 'Oracle vol-managed (net)')
    all_metrics.append(m_n)
    for m in (m, m_n):
        print(f"{m['label']:<26}"
              f"{m['total_return']*100:>8.1f}%"
              f"{m['annualized_return']*100:>8.1f}%"
              f"{m['annualized_vol']*100:>8.1f}%"
              f"{m['sharpe']:>8.2f}"
              f"{m['sortino']:>9.2f}"
              f"{m['max_drawdown']*100:>8.1f}%"
              f"{m['calmar']:>8.2f}")
    print("-" * 80)

    # Save metrics
    pd.DataFrame(all_metrics).to_csv(TBL / 'G1_strategy_metrics.csv', index=False)

    # ===== Plot equity curves =====
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    eq_bh = (1 + df_bh['r_gross']).cumprod()
    axes[0].plot(eq_bh.index, eq_bh.values, lw=1.8, color='#2b6cb0',
                  label='Buy-and-hold')
    colors = {'HAR-RS-DOW': '#dc2626', 'HAR-WE': '#f59e0b', 'HAR-RS-Q-WE-X': '#7c3aed'}
    for f in fams:
        eq = (1 + strats[f]['r_net']).cumprod()
        axes[0].plot(eq.index, eq.values, lw=1.4, color=colors[f],
                      label=f'{f} (net)')
    eq_oracle = (1 + df_oracle['r_net']).cumprod()
    axes[0].plot(eq_oracle.index, eq_oracle.values, lw=1.0, ls='--',
                  color='#059669', label='Oracle (net)')

    axes[0].set_ylabel('Cumulatieve return (€1 startbedrag)')
    axes[0].set_title('Vol-managed BTC-EUR vs Buy-and-hold — OOS')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # Position weights of winner
    axes[1].plot(strats['HAR-RS-DOW'].index, strats['HAR-RS-DOW']['weight'],
                  lw=0.7, color='#dc2626', label='HAR-RS-DOW weight')
    axes[1].axhline(1.0, color='k', linestyle=':', lw=0.5, label='unit weight')
    axes[1].axhline(MAX_LEVERAGE, color='red', linestyle=':', lw=0.5,
                     label='leverage cap')
    axes[1].set_ylabel('Position weight')
    axes[1].set_xlabel('Datum')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig.tight_layout()
    fig.savefig(FIG / 'G1_equity_curves.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {(FIG / 'G1_equity_curves.png').relative_to(PROJECT)}")

    # ===== Markdown summary =====
    md = ["# G1 — Economic value: vol-managed strategies\n",
           f"OOS periode: {dates.min().date()} → {dates.max().date()} ({len(dates)} dagen)\n",
           f"Target annual vol: {TARGET_VOL_ANNUAL*100:.0f}%, "
           f"max leverage: {MAX_LEVERAGE}×, "
           f"TC: {TC_RATE*100:.2f}% per zijde\n\n",
           "## Performance metrics\n",
           pd.DataFrame(all_metrics).to_markdown(index=False, floatfmt='.4f')]
    SUM.write_text("\n".join(md))
    print(f"Summary: {SUM.relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
