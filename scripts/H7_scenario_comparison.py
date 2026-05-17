"""H7 — Scenario-vergelijking: rendement vs risico over kapitaal-niveaus.

Testen we de drie scenario's uit de platform-discussie:
  A: Bitvavo spot vol-managed (current bot baseline)
  B: Bitvavo + DeGiro ETP mix (lagere fees, BTC-tracker exposure)
  C: Theoretical covered-call overlay (options-income proxy)

Plus baselines:
  Buy-and-hold (passief), pure-trend (geen vol-info), HAR-vol-target-only

Aan vier kapitaal-niveaus: €110, €1k, €5k, €25k.

ALLE backtests gebruiken DEZELFDE prijsdata en HAR-σ forecasts.
Verschil zit in fee-structuur en strategie-logica.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')

# === 1. Load shared data ===
df_har = pd.read_parquet(P / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
df_har.index = pd.to_datetime(df_har.index)
if df_har.index.tz is not None: df_har.index = df_har.index.tz_localize(None)
df_har['sigma_har'] = np.sqrt(np.exp(df_har['mu_pred'] + 0.5 * df_har['sigma_pred']**2))
df_har['sigma_har_ann'] = df_har['sigma_har'] * np.sqrt(365)

# BTC prices (afgeleid van returns)
df_har['S'] = 100 * (1 + df_har['r_actual']).cumprod()

# MA50 trend signal
df_har['ma50'] = df_har['S'].rolling(50, min_periods=50).mean()
df_har['in_trend'] = (df_har['S'] > df_har['ma50']).astype(int)

# ATR (14) for ATR-stop strategy
def true_range(df):
    tr = pd.concat([
        df['S'].diff().abs(),
        df['S'] - df['S'].shift(),
    ], axis=1).abs().max(axis=1)
    return tr

df_har['atr14'] = true_range(df_har).rolling(14, min_periods=14).mean()

# === 2. Fee structures per platform ===
FEES = {
    'bitvavo_spot':  {'maker': 0.0015, 'taker': 0.0025, 'min_fee': 0.0},
    'degiro_etp':    {'pct': 0.0,      'flat': 2.50,    'free_first': True},
    'mixed_ab':      {'maker': 0.0010, 'taker': 0.0015, 'min_fee': 0.5},  # blended
}

# === 3. Strategy implementations ===
def backtest(strategy_name, initial_capital, fee_model='bitvavo_spot',
             target_vol_ann=0.35, use_trend=True, use_atr_stop=False,
             covered_call_yield=0.0):
    """Generieke backtest met variabele strategy-knobs.
    
    Returns: dict met portfolio-curve, metrics, fees_paid.
    """
    cash = initial_capital
    btc = 0.0
    fees_total = 0.0
    n_trades = 0
    
    # Equity curve
    equity = pd.Series(index=df_har.index, dtype=float)
    target_w_series = pd.Series(index=df_har.index, dtype=float)
    current_stop = -np.inf  # voor ATR-stop strategie
    in_position = False
    
    fees = FEES[fee_model]
    
    for i, t in enumerate(df_har.index):
        row = df_har.loc[t]
        price = row['S']
        
        # Total wealth
        wealth = cash + btc * price
        equity.loc[t] = wealth
        
        # === Bepaal target weight w_t ===
        if strategy_name == 'buy_and_hold':
            target_w = 1.0 if i > 0 else 1.0
        
        elif strategy_name == 'pure_har_voltarget':
            # HAR vol-targeting alleen, geen trend
            if pd.isna(row['sigma_har_ann']):
                target_w = 0
            else:
                target_w = min(1.0, target_vol_ann / row['sigma_har_ann'])
        
        elif strategy_name == 'har_voltarget_trend':
            # HAR vol-target ALLEEN als in trend
            if row['in_trend'] == 1 and not pd.isna(row['sigma_har_ann']):
                target_w = min(1.0, target_vol_ann / row['sigma_har_ann'])
            else:
                target_w = 0
        
        elif strategy_name == 'atr_stop_trend':
            # ATR-stop + trend overlay (de winner uit H2)
            if pd.isna(row['atr14']) or row['atr14'] == 0:
                target_w = 0
            elif not in_position and row['in_trend'] == 1:
                # Enter
                target_w = 0.90
                current_stop = price - 3.0 * row['atr14']
                in_position = True
            elif in_position:
                # Update trailing stop
                current_stop = max(current_stop, price - 3.0 * row['atr14'])
                if price < current_stop or row['in_trend'] == 0:
                    target_w = 0
                    in_position = False
                else:
                    target_w = 0.90
            else:
                target_w = 0
        
        elif strategy_name == 'pure_trend_ma50':
            target_w = row['in_trend'] if not pd.isna(row['in_trend']) else 0
        
        elif strategy_name == 'mixed_btc_etp':
            # Scenario B: 60% in DeGiro ETP (buy-hold), 40% Bitvavo vol-target
            # Buy-hold 60% blijft full exposure altijd
            if row['in_trend'] == 1 and not pd.isna(row['sigma_har_ann']):
                bitvavo_part = 0.40 * min(1.0, target_vol_ann / row['sigma_har_ann'])
            else:
                bitvavo_part = 0
            target_w = 0.60 + bitvavo_part
        
        elif strategy_name == 'covered_call_overlay':
            # Scenario C proxy: spot exposure + maandelijkse premium-yield uit covered calls
            # Theoretical: assume covered_call_yield (annualized) verdiend als constante drift
            target_w = 0.90 if row['in_trend'] == 1 else 0.30
            # Bonus rendement via "premium yield" — pas hieronder toe als drift
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        target_w_series.loc[t] = target_w
        
        # === Rebalance toward target weight ===
        if i == 0 or pd.isna(target_w):
            continue
        
        target_btc_value = wealth * target_w
        delta_btc_value = target_btc_value - btc * price
        
        # Rebalance threshold: alleen handelen bij >2% deviation om fees te besparen
        if abs(delta_btc_value) / wealth > 0.02:
            if fee_model == 'degiro_etp':
                # DeGiro: flat €2.50 per trade
                fee = fees['flat']
            elif fee_model == 'mixed_ab':
                # Geblende fee
                fee = abs(delta_btc_value) * fees['maker']
            else:
                # Bitvavo
                fee = abs(delta_btc_value) * fees['maker']  # assume maker (post-only)
                if fee < fees['min_fee']:
                    fee = fees['min_fee']
            
            # Apply trade
            if delta_btc_value > 0:
                # Buy BTC: cash → btc
                buyable = (delta_btc_value - fee) / price
                btc += buyable
                cash -= delta_btc_value
            else:
                # Sell BTC
                sellable = -delta_btc_value / price
                btc -= sellable
                cash += (-delta_btc_value - fee)
            
            cash -= fee
            fees_total += fee
            n_trades += 1
        
        # === Apply covered-call premium yield (alleen voor scenario C) ===
        if strategy_name == 'covered_call_overlay' and covered_call_yield > 0:
            # Premium = X% per year, op de waarde van de BTC-positie
            daily_premium = (covered_call_yield / 365) * btc * price
            cash += daily_premium
    
    # Final equity = initial * compounded daily returns
    equity = equity.dropna()
    
    # Metrics
    log_ret = np.log(equity / equity.shift(1)).dropna()
    final = equity.iloc[-1]
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    
    metrics = {
        'strategy': strategy_name,
        'fee_model': fee_model,
        'initial': initial_capital,
        'final': final,
        'total_return': (final / initial_capital - 1) * 100,
        'annualized_return': ((final / initial_capital) ** (1/n_years) - 1) * 100,
        'volatility_ann': log_ret.std() * np.sqrt(365) * 100,
        'sharpe': (log_ret.mean() * 365) / (log_ret.std() * np.sqrt(365)) if log_ret.std() > 0 else 0,
        'sortino': (log_ret.mean() * 365) / (log_ret[log_ret < 0].std() * np.sqrt(365)) if (log_ret < 0).any() and log_ret[log_ret < 0].std() > 0 else np.nan,
        'mdd_pct': ((equity / equity.cummax()) - 1).min() * 100,
        'calmar': ((final/initial_capital)**(1/n_years) - 1) / abs(((equity/equity.cummax())-1).min()) if ((equity/equity.cummax())-1).min() < 0 else np.nan,
        'fees_paid': fees_total,
        'fees_pct': (fees_total / initial_capital) * 100,
        'n_trades': n_trades,
        'equity': equity,
    }
    return metrics


# === 4. Run alle scenario's × kapitaal-niveaus ===
SCENARIOS = [
    # name, fee_model, params
    ('A: Buy-and-hold',                'bitvavo_spot', dict(strategy_name='buy_and_hold')),
    ('A: Pure HAR vol-target',         'bitvavo_spot', dict(strategy_name='pure_har_voltarget')),
    ('A: HAR vol-target + MA50',       'bitvavo_spot', dict(strategy_name='har_voltarget_trend')),
    ('A: ATR-stop + MA50 (current bot)','bitvavo_spot', dict(strategy_name='atr_stop_trend')),
    ('A: Pure trend MA50',             'bitvavo_spot', dict(strategy_name='pure_trend_ma50')),
    ('B: Mix Bitvavo + DeGiro ETP',    'mixed_ab',     dict(strategy_name='mixed_btc_etp')),
    ('C: Covered-call overlay (theor.)','bitvavo_spot', dict(strategy_name='covered_call_overlay', 
                                                              covered_call_yield=0.12)),  # 12%/year proxy
]

CAPITAL_LEVELS = [110, 1_000, 5_000, 25_000]

all_results = []
for cap in CAPITAL_LEVELS:
    print(f"\n{'='*100}")
    print(f"KAPITAAL-NIVEAU: €{cap:,}")
    print(f"{'='*100}")
    print(f"{'Scenario':<40}{'Sharpe':>8}{'Sortino':>9}{'MDD%':>8}{'AnnRet%':>10}{'Fees%':>8}{'Final€':>10}{'#Trades':>8}")
    print("-"*100)
    
    for name, fee_model, params in SCENARIOS:
        r = backtest(initial_capital=cap, fee_model=fee_model, **params)
        r['scenario_name'] = name
        r['capital_level'] = cap
        all_results.append(r)
        
        sortino_str = f"{r['sortino']:+.2f}" if not np.isnan(r['sortino']) else "  n/a"
        print(f"{name:<40}{r['sharpe']:+8.2f}{sortino_str:>9}{r['mdd_pct']:>8.1f}{r['annualized_return']:>10.1f}"
              f"{r['fees_pct']:>8.1f}{r['final']:>10.0f}{r['n_trades']:>8}")

# === 5. Save resultaten als dataframe ===
df_results = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'equity'} 
    for r in all_results
])
df_results.to_csv(P/'outputs/tables/H7_scenario_comparison.csv', index=False)

# === 6. Plots: equity curves per kapitaal-niveau ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, cap in zip(axes.flat, CAPITAL_LEVELS):
    scenarios_at_cap = [r for r in all_results if r['capital_level'] == cap]
    for r in scenarios_at_cap:
        equity_norm = r['equity'] / r['initial']
        ax.plot(equity_norm.index, equity_norm.values,
                label=f"{r['scenario_name'].split(':')[1].strip() if ':' in r['scenario_name'] else r['scenario_name']} (Sh={r['sharpe']:+.2f})",
                lw=1.2, alpha=0.85)
    ax.set_title(f'Initial capital: €{cap:,}', fontweight='bold')
    ax.set_ylabel('Wealth / Initial')
    ax.axhline(1, color='gray', ls='--', lw=0.5)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
fig.suptitle('H7 — Scenario comparison: rendement vs risico over kapitaal-niveaus',
              fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(P/'outputs/figures/H7_scenario_comparison.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Saved: outputs/tables/H7_scenario_comparison.csv")
print(f"✓ Figuur: outputs/figures/H7_scenario_comparison.png")
