# HAR-RS-DOW: Variance Forecasting for BTC-EUR

> **MSc Econometrics & Operations Research thesis** — Financial Track, Vrije Universiteit Amsterdam (2025-2026)
> An econometric study establishing HAR-RS-DOW as the statistically dominant variance-forecaster for BTC-EUR daily realized variance, with canonical applications to risk management (VaR / Expected Shortfall) and option pricing (Black-Scholes).

[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10+-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Paper contribution — three claims

The paper establishes three connected empirical claims about HAR-RS-DOW for BTC-EUR:

### Claim 1 — Statistical dominance (in-sample and out-of-sample)

HAR-RS-DOW is the empirically best variance-forecaster for BTC-EUR daily realized variance, dominating nine HAR-family alternatives on every metric tested:

| Metric | HAR-RS-DOW | Best alternative | Margin |
|---|---|---|---|
| In-sample BIC | -10 462 | -9 972 | ΔBIC = 490 (Bayes factor ~10¹⁰⁶) |
| OOS CRPS (1 312 days) | **0.498** | 0.582 | -14% (HAR-RS-Q-WE-X) |
| OOS R² (log-RV) | **+0.439** | +0.246 | +76% relative |
| Diebold-Mariano vs four alternatives | dominant | — | p < 0.005 across all |
| Multi-horizon h = 5 dominance | DM = **-5.46** | — | p < 10⁻⁷ |

The semivariance decomposition (β_d⁻ = 0.34 vs β_d⁺ = 0.18) confirms the leverage effect at the variance level for crypto, and the day-of-week dummies capture systematic weekly seasonality (γ_saturday = -0.281, γ_monday = +0.142).

### Claim 2 — Adequacy for Basel III risk forecasting

HAR-driven density forecasts produce **conservatively calibrated** Value-at-Risk and Expected Shortfall across three density specifications. The Acerbi-Szekely (2014) Z1 statistic is positive for all three:

| Density assumption | ES α=1% forecast | ES α=5% forecast | Z1 statistic | Verdict |
|---|---|---|---|---|
| Normal | -5.93% | -4.59% | +2.11 / +1.90 | Conservative — capital-adequate |
| Student-t (ν=3) | -8.99% | -4.97% | +1.73 / +1.83 | Conservative |
| Hansen skewed-t (ν=4.4, λ=+0.01) | -7.86% | -5.06% | +1.83 / +1.81 | Conservative |
| Realized ES (1 312 OOS days) | -5.84% | -3.91% | — | — |

Z1 > 0 implies the model **over-reserves** capital — a desirable property for regulatory adequacy under FRTB.

### Claim 3 — Edge in option pricing

Using HAR-σ̂ as input to Black-Scholes for 7-day at-the-money straddles produces fair-value pricing closer to market-clearing than naive vol-input alternatives:

| σ-model | Avg premium | Avg payoff | Edge | Hit rate |
|---|---|---|---|---|
| Constant σ (sample mean) | €11.16 | €10.30 | +€0.86 | 66.4% over-priced |
| Rolling 30-day σ | €10.59 | €10.30 | +€0.29 | 63.0% |
| **HAR-RS-DOW σ̂** | **€10.50** | €10.30 | **+€0.20** | **60.0%** |

The HAR-driven price is closest to fair value — operationally relevant for any market-maker quoting BTC options on platforms like Deribit, where consistent fair-value quotes capture flow without leaving systematic edge to counterparties.

---

## Why this matters

HAR-RS-DOW is, by construction, a **variance forecasting model** — it predicts the *magnitude* of future return variation conditional on heterogeneous-horizon information in past realized variance, semivariance decomposition, and day-of-week effects. The paper evaluates whether this statistical edge translates into the applications where variance forecasts are the right input:

- **VaR + Expected Shortfall** under Basel III FRTB: yes, conservatively calibrated
- **Black-Scholes option pricing**: yes, minimum-edge fair-value pricing
- **Density forecasting** for risk-budgeting and stress-testing: yes, PIT-calibrated

The paper explicitly does **not** position HAR-RS-DOW as a directional trading signal — that would conflate two distinct econometric problems (variance vs. expected return).

---

## Paper structure

The full thesis report is in [`outputs/thesis_progress_report.pdf`](outputs/thesis_progress_report.pdf), structured in three parts:

- **Deel I** — HAR-RS-DOW as variance forecaster (statistical contribution)
- **Deel II** — Risk management applications (VaR, Expected Shortfall, density-based)
- **Deel III** — Discussion, methodological role, limitations

A pedagogical walkthrough is available in [`outputs/thesis_explained_econometrically.md`](outputs/thesis_explained_econometrically.md) and as Overleaf-ready LaTeX in [`outputs/latex/`](outputs/latex/).

---

## Data and replication

OOS evaluation window: 2022-10-01 → 2026-05-14 (1 312 days). Estimation sample begins 2019-04-03 (Bitvavo BTC-EUR market inception plus 26-day warmup for realized measures).

```bash
pip install -r requirements.txt
pytest tests/test_reproducibility.py -v    # 20 passing tests, ~2s
```

Test coverage spans data integrity, model performance (CRPS=0.498, QLIKE=0.595, R²=+0.44), DM-dominance over alternatives, ES Z1 conservatism, Hansen skewed-t calibration, mathematical correctness (Black-Scholes put-call parity, Normal ES, t(3) ES heavy-tail behavior).

Raw OHLC data (~186 MB) is excluded from the repo and fetched via the Bitvavo public REST API:

```bash
python scripts/A1_fetch_bitvavo_candles.py
python scripts/B2_recompute_realized_v2.py
```

Period covered: 2019-03-08 → 2026-05-15 at five frequencies (1m, 5m, 15m, 1h, 1d), plus ETH-EUR for cross-asset robustness checks.

---

## Repository layout

```
crypto_hartrace/
├── src/hartrace/           # Econometric library
│   ├── distributions.py    # Hansen 1994 skewed-t implementation
│   ├── estimation.py       # HAR-family MLE
│   ├── features_v2.py      # Daily realized variance features
│   └── external_data.py    # FRED + yfinance integration
├── scripts/                # Empirical pipeline
│   ├── B1, C1              # Signature plot + stylized facts
│   ├── D1                  # Horse race (10 HAR variants)
│   ├── E1, E2, E4b         # Walk-forward, VaR coverage, multi-horizon
│   ├── H4                  # Expected Shortfall (Acerbi-Szekely Z1)
│   ├── H5                  # Density specification robustness
│   └── H6                  # Option pricing simulation
├── tests/                  # Reproducibility test suite (20 tests)
├── outputs/
│   ├── thesis_progress_report.{md,pdf}
│   ├── thesis_explained_econometrically.md   # Pedagogical walkthrough
│   ├── latex/              # Overleaf-ready LaTeX
│   ├── tables/             # CSV / parquet result files
│   ├── figures/            # PNG plots
│   └── presentation/       # 22-slide defense deck
└── docs/                   # Auxiliary documentation
```

---

## Engineering note: live trading implementation

In addition to the paper analyses, this repository contains an engineering implementation of an automated trading bot deployed on Bitvavo. **This is not part of the paper's central claim** — HAR-RS-DOW is not positioned as a directional trading signal. The trading code serves two secondary purposes:

1. **Production validation** of the data pipeline and HAR estimation routines — proves the model can run end-to-end in real-time
2. **Empirical scope check** — investigates under what conditions vol-information adds economic value (answer: position-sizing within trend overlay, modest improvement)

The trading bot strategy converged empirically on simple trend-following (MA50 daily) with skip-weekend execution timing. The skip-weekend filter is itself derived from HAR-RS-DOW's DOW seasonality (γ_saturday = -0.281, indicating systematically lower weekend volatility and consequently wider bid-ask spreads). All code is in `src/hartrace/live/` with documentation in `docs/PAPER_TO_LIVE_RUNBOOK.md`.

The implementation is not a recommendation. Trading cryptocurrency carries substantial risk; the live bot operates with €110 of personal capital purely as an engineering experiment.

---

## Citation

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

---

## Theoretical foundations

- **Corsi, F. (2009)**. *A simple approximate long-memory model of realized volatility*. Journal of Financial Econometrics, 7(2), 174–196.
- **Patton, A. J., & Sheppard, K. (2015)**. *Good volatility, bad volatility: Signed jumps and the persistence of volatility*. Review of Economics and Statistics, 97(3), 683–697.
- **Hansen, B. E. (1994)**. *Autoregressive conditional density estimation*. International Economic Review, 35(3), 705–730.
- **Acerbi, C., & Szekely, B. (2014)**. *Back-testing expected shortfall*. Risk, 27(11), 76–81.
- **Diebold, F. X., & Mariano, R. S. (1995)**. *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3), 253–263.
- **Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008)**. *Designing realised kernels to measure the ex-post variation of equity prices*. Econometrica, 76(6), 1481–1536.
- **Basel Committee on Banking Supervision (2019)**. *Minimum capital requirements for market risk*. Bank for International Settlements.

Full reference list in [`outputs/thesis_progress_report.md`](outputs/thesis_progress_report.md).

---

## License

MIT — see [LICENSE](LICENSE). The paper analyses and code are released for academic and educational use.

## Contact

Sake Saakstra — MSc EOR Financial Track, Vrije Universiteit Amsterdam
