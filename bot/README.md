# Trading bot — engineering implementation

> **This directory documents the live trading bot.**
> The bot is an engineering artifact of this project, NOT part of the paper's contribution.
> See [`../paper/paper.md`](../paper/paper.md) for the academic work on HAR-RS-DOW.

---

## Status

A daily trading bot is deployed to Bitvavo with €110 of personal capital, running pure trend MA50 (current strategy after empirical convergence) with skip-weekend filter. The bot is **personal engineering**, not a recommendation or financial advice. Cryptocurrency trading carries substantial risk.

## Why a bot exists in this repository

The bot serves two engineering purposes connected to (but not part of) the paper:

1. **Production validation of the data pipeline** — the same Bitvavo data-fetch, realized-variance computation, and HAR estimation routines used in the paper are exercised daily in production. This proves the model can run end-to-end in real-time.
2. **Empirical scope check on HAR's economic value** — H7 through H11 (see below) investigate whether HAR variance forecasts add directional trading alpha. The answer is **no**: HAR's variance-edge does not translate into trading PnL beyond what simple trend-following already captures. This finding empirically supports the paper's central position that HAR is a variance forecaster, not a directional signal.

## Bot architecture

Located in `../src/hartrace/live/`:

| File | Purpose |
|---|---|
| `forecaster.py` | Produces daily HAR-σ̂ forecasts from realized measures |
| `bitvavo_client.py` | Authenticated Bitvavo REST client (HMAC, tick-size rounding) |
| `executor.py` | Decision loop: kill-switch → skip-weekend → strategy → safety stack → order |
| `safety.py` | Three-layer safety: env flags + CLI flag + kill-switch file |
| `trend_filter.py` | Pure trend MA50 signal (current default strategy) |
| `atr_filter.py` | ATR-stop + trend (legacy strategy, available via `STRATEGY=atr_trend`) |
| `config.py` | Typed configuration with env-variable overrides |

Entry point: `../scripts/btc_live_trader.py`
Scheduled run: `../launchd/com.sakesaakstra.btc-trader.plist` (daily 22:05 UTC)

## Strategy R&D scripts (`../scripts/H7` – `../scripts/H11`)

| Script | Question | Finding |
|---|---|---|
| `H7_scenario_comparison.py` | Which trading scenario fits which capital scale? | Pure trend MA50 best on Bitvavo spot |
| `H8_robustness_extensions.py` | Is the strategy robust across MA-windows / stress periods / walk-forward selection? | MA50 is optimal across all dimensions |
| `H9_frequency_test.py` | What execution frequency is optimal? | Daily — every higher frequency destroys capital via fees |
| `H10_timing_dow_test.py` | Cron-time + DOW filtering? | Skip-weekend filter from HAR-RS-DOW γ_sat = −0.281 adds +0.06 Sharpe, −25% fees |
| `H11_har_execution_filters.py` | Do other HAR-RS-DOW outputs add execution value? | No — confirms HAR is for variance, not trading-alpha |

All R&D scripts produce CSV tables in `../outputs/tables/H*.csv` and figures in `../outputs/figures/H*.png` for reproducibility.

## Operator runbook

See [`../docs/PAPER_TO_LIVE_RUNBOOK.md`](../docs/PAPER_TO_LIVE_RUNBOOK.md) for:

- Setup of `~/W8W.env` with Bitvavo API credentials
- Daily monitoring procedures
- Kill-switch usage (`touch ~/STOP_TRADING`)
- Strategy switching via environment variables
- Rollback procedures

## Safety architecture

Three independent layers must align before any live order is placed:

1. `LIVE_TRADING=true` in `~/W8W.env` (explicit env flag)
2. `DRY_RUN=false` in `~/W8W.env` (explicit env flag)
3. `--live` CLI flag passed by launchd (explicit operator intent)

In addition, `~/STOP_TRADING` file existence blocks all order placement regardless of other settings. Any layer breaking blocks orders. Default state is paper-mode (no orders).

## Important separation from the paper

The paper at [`../paper/paper.md`](../paper/paper.md) does **not**:

- Position HAR-RS-DOW as a directional trading signal
- Claim trading-PnL results
- Reference the live bot deployment as a contribution

The paper does:

- Establish HAR-RS-DOW as the statistically dominant variance forecaster
- Demonstrate Basel III Expected Shortfall adequacy
- Demonstrate Black-Scholes option-pricing fair-value performance

The bot is downstream engineering. It depends on the paper's variance-forecasting infrastructure but is not itself part of the academic contribution.
