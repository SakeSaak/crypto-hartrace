# HAR-RS-DOW: Variance Forecasting and Risk Management for BTC-EUR

> **MSc Econometrics & Operations Research thesis** — Financial Track, Vrije Universiteit Amsterdam (2025-2026)
> Empirical study of HAR-family models as variance forecasters for BTC-EUR, with the canonical applications: Value-at-Risk, Expected Shortfall (Basel III mandated), option pricing, and density-based scenario analysis.

[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10+-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## TL;DR — what's in this repo?

A complete end-to-end research pipeline that:

1. **Estimates a HAR-RS-DOW model** for BTC-EUR daily realized variance — the winner of an in-sample horse race over nine HAR-family alternatives (ΔBIC = 490 over runner-up; OOS CRPS = 0.498; R²_oos = +0.44; Diebold-Mariano dominance with p < 0.005 vs. all four nearest competitors)
2. **Applies the variance forecasts to risk management** — the canonical domain for vol-forecasters: Value-at-Risk (Christoffersen-calibrated), Expected Shortfall (Acerbi-Szekely Z1 ∈ [+1.73, +2.11], conservative across Normal, Student-t and Hansen 1994 skewed-t densities), and density-based scenario analysis
3. **Demonstrates an options-pricing application** — HAR-σ as input to Black-Scholes produces fair-value pricing closer to market clearing than constant- or rolling-window σ (edge €0.20 vs €0.86 / €0.29)
4. **Implements the live-deployment infrastructure** — a Bitvavo trading executor with three-layer safety, demonstrating the gap between backtest and production via two encountered bugs (tick-size rounding, HMAC signature for GET-with-query)

Full thesis report (PDF): [`outputs/thesis_progress_report.pdf`](outputs/thesis_progress_report.pdf)

## Key empirical findings

### Statistical performance (variance forecasting)

| Metric | HAR-RS-DOW | Next-best baseline | Margin |
|---|---|---|---|
| In-sample BIC | -10 462 | -9 972 | ΔBIC = 490 (Bayes factor ~10¹⁰⁶) |
| OOS CRPS (1 312 days) | 0.498 | 0.582 | -14% |
| OOS R² (log-RV) | +0.44 | +0.25 | +76% relative |
| DM-test vs. four nearest alternatives | dominant | — | p < 0.005 on all |
| Multi-horizon h = 5 dominance | DM = -5.46 | — | p < 10⁻⁷ |

### Risk-management calibration (the canonical HAR application)

| Density assumption | ES α=1% | ES α=5% | Z1 statistic | Verdict |
|---|---|---|---|---|
| Normal | -5.93% | -4.59% | +2.11 / +1.90 | Conservative — Basel III adequate |
| Student-t (ν=3) | -8.99% | -4.97% | +1.73 / +1.83 | Conservative |
| Hansen skewed-t (ν=4.4, λ=+0.01) | -7.86% | -5.06% | +1.83 / +1.81 | Conservative |
| Realized (1 312 OOS days) | -5.84% | -3.91% | — | — |

Acerbi-Szekely (2014) Z1 > 0 across all three densities → HAR-driven ES forecasts are robust to density specification.

## Why this matters

HAR-RS-DOW is, by construction, a **variance forecasting model** — it predicts the *magnitude* of future return variation, conditional on the heterogeneous-horizon information in past realized variance. This thesis evaluates whether that statistical edge translates into **the applications where variance forecasts are the right primary input**:

- **Value-at-Risk and Expected Shortfall** (Basel III mandated risk measures): HAR produces conservatively-calibrated ES across three density specifications (Normal, Student-t, Hansen 1994 skewed-t), with Acerbi-Szekely Z1 ∈ [+1.73, +2.11]. Adequate for regulatory capital reservation.
- **Option pricing**: HAR-σ as input to Black-Scholes produces minimum-edge fair-value pricing (€0.20 average mispricing vs €0.86 for constant σ and €0.29 for rolling-window σ). Material edge for any market-maker quoting derivatives.
- **Density forecasting**: PIT-calibrated probabilistic predictions for risk-budgeting, stress testing, and scenario analysis.

The thesis does **not** position HAR as a directional trading signal — that would conflate two distinct econometric problems (variance vs. expected return). A small set of trading experiments is included in the appendix as a *scope-of-applicability* check, but the core contribution is the variance-forecasting framework and its risk-management consequences.

## Architecture

```
crypto_hartrace/
├── src/hartrace/           # Core econometric library
│   ├── distributions.py    # Hansen 1994 skewed-t implementation
│   ├── estimation.py       # HAR-family MLE
│   ├── features_v2.py      # Daily realized variance features
│   ├── external_data.py    # FRED + yfinance integration
│   └── live/               # Production trading module
│       ├── atr_filter.py     # ATR-trailing stop strategy
│       ├── bitvavo_client.py # HMAC-authenticated REST client
│       ├── config.py         # Three-layer safety config
│       ├── executor.py       # Trade orchestration
│       ├── forecaster.py     # HAR forecast wrapper
│       └── safety.py         # Safety gates + kill switch
├── scripts/                # Pipeline analyses
│   ├── B1, C1               # Signature plot + stylized facts
│   ├── D1                   # Model horse race
│   ├── E1, E2, E4b          # Walk-forward + VaR + multi-horizon
│   ├── E2, H4               # VaR + Expected Shortfall (canonical applications)
│   ├── H5                   # Density-specification robustness check
│   ├── H6                   # Option-pricing simulation (Black-Scholes with HAR-σ)
│   ├── G1-G5, H1-H3         # Scope-of-applicability checks (trading)
│   ├── btc_live_trader.py   # Live entry point
│   └── build_defense_deck.py
├── tests/test_reproducibility.py  # 20 passing tests
├── outputs/
│   ├── tables/              # CSV/parquet results
│   ├── figures/             # PNG plots
│   ├── thesis_progress_report.{md,pdf}
│   ├── thesis_explained_econometrically.md  # Pedagogical walkthrough
│   └── presentation/        # 22-slide defense deck
└── docs/                   # Operator runbooks
```

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite
pytest tests/test_reproducibility.py -v
# Expected: 20 passed in ~2s
```

Test coverage:
- Data integrity (BTC/ETH candles, walk-forward parquets)
- Model performance (CRPS, QLIKE, R², DM dominance)
- Risk forecasting (ES Z1 conservatism, Hansen skewed-t calibration)
- Live trading components (ATR filter logic, Bitvavo tick rounding)
- Mathematical correctness (Normal ES, t(3) ES, Black-Scholes put-call parity)
- Strategy backtests (ATR-stop winner, HAR-voltarget top-3)

## Data

Raw OHLC data is fetched from the Bitvavo public REST API and is excluded from this repo (~186 MB). To reproduce the data layer:

```bash
python scripts/A1_fetch_bitvavo_candles.py  # Re-downloads candles
python scripts/B2_recompute_realized_v2.py  # Rebuilds realized measures
```

Period covered: 2019-03-08 → 2026-05-15 for BTC-EUR (1m, 5m, 15m, 1h, 1d) plus ETH-EUR. Out-of-sample evaluation window: 2022-10-01 → 2026-05-14 (1 312 days).

## Live deployment (honest disclosure)

The bot is deployed in production on Bitvavo with three-layer safety. Two bugs found and fixed in the first live trade:

1. **Tick-size violation**: Bitvavo BTC-EUR requires whole-euro prices (tickSize=€1.00). Fixed via per-side `_round_price_to_tick()` floor/ceil rounding.
2. **HMAC signature for GET with query string**: Bitvavo's HMAC requires the signature-path to include the query string. Fixed via `urlencode(params)` in the signature builder.

Full operator runbook: [`docs/PAPER_TO_LIVE_RUNBOOK.md`](docs/PAPER_TO_LIVE_RUNBOOK.md)

This is a deliberate research-honesty choice: most academic papers claim "live-ready" without ever placing a real order. The implementation gap matters.

## Citation

If you use ideas, code, or results from this repository in academic work:

```bibtex
@mastersthesis{saakstra2026harvar,
  author  = {Saakstra, Sake},
  title   = {Variance Forecasting and Risk Management for {BTC-EUR}
             using {HAR-RS-DOW} Models},
  school  = {Vrije Universiteit Amsterdam},
  year    = {2026},
  type    = {Paper},
  note    = {{MSc} Econometrics \& Operations Research, Financial Track}
}
```

## Theoretical foundations

The HAR family builds on three key papers:

- **Corsi, F. (2009)**. *A simple approximate long-memory model of realized volatility*. Journal of Financial Econometrics, 7(2), 174–196.
- **Patton, A. J., & Sheppard, K. (2015)**. *Good volatility, bad volatility: Signed jumps and the persistence of volatility*. Review of Economics and Statistics, 97(3), 683–697.
- **Hansen, B. E. (1994)**. *Autoregressive conditional density estimation*. International Economic Review, 35(3), 705–730.
- **Acerbi, C., & Szekely, B. (2014)**. *Back-testing expected shortfall*. Risk, 27(11), 76–81.

Full reference list in [`outputs/thesis_progress_report.md#16`](outputs/thesis_progress_report.md).

## License

MIT — see [LICENSE](LICENSE). **Risk disclaimer**: this code performs financial trading. Use at your own risk; the author makes no representations about suitability for production deployment in your environment.

## Contact

Sake Saakstra — MSc EOR Financial Track, Vrije Universiteit Amsterdam
