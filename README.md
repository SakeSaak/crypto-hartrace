# HAR-RS-DOW: Variance Forecasting and Risk Management for BTC-EUR

> **MSc Econometrics & Operations Research thesis** — Financial Track, Vrije Universiteit Amsterdam (2025-2026)
> Empirical study of HAR-family models for cryptocurrency volatility forecasting, with applications to Value-at-Risk, Expected Shortfall (Basel III), option pricing, and vol-managed trading.

[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10+-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## TL;DR — what's in this repo?

A complete end-to-end research pipeline that:

1. **Estimates a HAR-RS-DOW model** for BTC-EUR realized variance (winner of horse race over 9 alternatives — CRPS = 0.498 OOS, R²_oos = +0.44, DM-dominance p < 0.005)
2. **Applies the variance forecasts** to risk management (Expected Shortfall Z1 = +1.73 conservative across Normal, Student-t, and Hansen skewed-t densities), option pricing (HAR-σ produces fair-value pricing closest to market clearing), and trading (HAR vol-target + trend overlay → Sharpe 0.94 ROBUST)
3. **Deploys live on Bitvavo** with three-layer safety, ATR-trailing stops, and honest disclosure of two production bugs caught in the first live trade

Full thesis report (PDF): [`outputs/thesis_progress_report.pdf`](outputs/thesis_progress_report.pdf)

## Key empirical findings

| Result | Value | Verdict |
|---|---|---|
| HAR-RS-DOW vs runner-up (BIC) | ΔBIC = 490 | Bayes factor ~10¹⁰⁶ |
| OOS CRPS over 1 312 days | 0.498 | vs 0.582 next best (-14%) |
| Diebold-Mariano vs all alternatives | DM ∈ [-7.5, -6.9] | p < 0.005 |
| Expected Shortfall Z1 (Acerbi-Szekely) | +1.73 to +2.11 | Conservative for Basel III |
| HAR vol-target + trend overlay Sharpe | 0.94 ROBUST | OOS, sub-period stable |
| ATR-stop + trend MA50 Sharpe (benchmark) | 1.14 | Marginally beats HAR |

## Why this matters

The thesis decomposes a long-standing empirical question: **HAR is statistically dominant, but where does that translate to economic value?** The answer is nuanced and matters:

- ✅ **Risk management**: HAR's variance edge produces Basel III-adequate VaR + ES forecasts
- ✅ **Option pricing**: HAR-σ produces minimum-edge fair-value pricing (€0.20 vs €0.86 for constant σ)
- ✅ **Position sizing within trend overlay**: HAR vol-target + MA50 is ROBUST across regimes
- ❌ **Pure directional trading**: HAR predicts magnitude, not direction; simple trend signals dominate
- ❌ **Retail HFT**: at <daily frequencies on €110 capital, fees destroy any edge

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
│   ├── G1-G5                # Trading evaluation
│   ├── H1-H6                # Multi-frequency, TA, robustness, ES, density, options
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
  type    = {{MSc} thesis},
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
