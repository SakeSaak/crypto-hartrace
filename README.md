# HAR-RS-DOW variance forecasting for BTC-EUR with Basel III ES and Black-Scholes applications

> **MSc Econometrics & Operations Research thesis project**, Vrije Universiteit Amsterdam (2025-2026)

This repository contains two distinct artifacts that share a codebase but have different status:

1. **The paper** ([`paper/paper.md`](paper/paper.md), [`paper/paper.pdf`](paper/paper.pdf)) — the academic contribution: HAR-RS-DOW as the statistically dominant variance forecaster for BTC-EUR, with canonical applications to risk management and option pricing.
2. **The trading bot** ([`bot/`](bot/) directory) — a personal engineering implementation that runs daily on Bitvavo with €110 of capital. **The bot is not part of the paper's contribution.**

Readers interested in the academic work should go to [`paper/paper.md`](paper/paper.md). Readers interested in the engineering implementation can consult [`bot/README.md`](bot/README.md).

---

## The paper at a glance

**Title:** HAR-RS-DOW Variance Forecasting for Bitcoin: Statistical Dominance and Applications in Risk Management and Option Pricing

The paper makes three connected empirical contributions for BTC-EUR daily realized variance over the period 3 April 2019 to 14 May 2026:

| # | Contribution | Key statistic |
|---|---|---|
| 1 | HAR-RS-DOW is the statistically dominant variance forecaster within the HAR family | In-sample ΔBIC = 490 over runner-up (Bayes factor ≈ 10¹⁰⁶); OOS CRPS = 0.498 vs 0.582 (best alt); DM-stat rejects equal predictive accuracy at p < 0.005 against all four nearest competitors |
| 2 | HAR-driven density forecasts are conservatively calibrated for Basel III risk-management | Acerbi-Szekely Z1 ∈ [+1.73, +2.11] across three density specifications (Normal, t(3), Hansen skewed-t) and both significance levels (α ∈ {1%, 5%}) |
| 3 | HAR-σ̂ as input to Black-Scholes yields fair-value option pricing | Avg edge €0.20 for 7-day ATM straddle (n=1 305 OOS valuations), against €0.86 for constant-σ and €0.29 for rolling-σ baselines |

**Sample:** 2 334 days estimation panel (3 April 2019 – 14 May 2026), with 1 312-day out-of-sample window from 1 October 2022 onwards. Walk-forward re-estimation every 20 days.

**Methodology:** Realized Kernel at 5-minute frequency (Barndorff-Nielsen et al. 2008); ten HAR-family specifications estimated by OLS with Newey-West HAC standard errors; conditional density via Hansen (1994) skewed-t for risk applications; CRPS and Diebold-Mariano (1995) testing for forecast comparison; Acerbi-Szekely (2014) Z1 for Expected Shortfall backtesting.

---

## Repository structure

```
crypto_hartrace/
├── paper/                     # Academic work — START HERE
│   ├── paper.md               # Full paper (Markdown source)
│   ├── paper.pdf              # Compiled PDF (working paper format)
│   └── _archive_pre_academic.md   # Pre-restructure draft, for diff
│
├── bot/                       # Trading bot — engineering, not paper
│   └── README.md              # Bot documentation (separate from paper)
│
├── src/hartrace/              # Econometric library (used by both paper & bot)
│   ├── distributions.py       # Hansen 1994 skewed-t implementation
│   ├── estimation.py          # HAR-family MLE
│   ├── features_v2.py         # Daily realized variance features
│   ├── external_data.py       # FRED + yfinance integration
│   └── live/                  # Live-bot subpackage (see bot/README.md)
│
├── scripts/
│   ├── B1, C1, D1             # Stylized facts + horse race (paper)
│   ├── E1, E2, E4b            # OOS walk-forward, VaR coverage, multi-horizon (paper)
│   ├── H4, H5, H6             # Expected Shortfall, density-robustness, option pricing (paper)
│   ├── H7 – H11               # Trading R&D — see bot/README.md
│   └── btc_live_trader.py     # Live bot entry point (see bot/README.md)
│
├── tests/                     # Reproducibility suite (20 passing tests, ~2s)
├── docs/
├── outputs/                   # Tables, figures, presentation, LaTeX explainer
└── data/                      # Excluded from git (~186 MB); fetched via Bitvavo REST API
```

---

## Reproducing the paper

```bash
pip install -r requirements.txt
pytest tests/test_reproducibility.py -v        # 20 tests, ~2s

# Re-run the empirical pipeline (data → realized measures → estimation → applications)
python scripts/A1_fetch_bitvavo_candles.py     # Pull OHLC data (~186 MB)
python scripts/B2_recompute_realized_v2.py     # Compute daily RK at 5-min frequency
python scripts/D1_horse_race.py                # In-sample model comparison (Table 1)
python scripts/E1_walkforward.py               # OOS walk-forward (Table 2)
python scripts/E4b_multihorizon.py             # Multi-horizon (Table 3)
python scripts/H4_expected_shortfall.py        # ES with Z1 backtest (Table 4)
python scripts/H5_density_robustness.py        # Density-spec robustness
python scripts/H6_option_pricing.py            # Black-Scholes valuation (Table 5)
```

All script outputs land in `outputs/tables/` (CSV/parquet) and `outputs/figures/` (PNG).

---

## Citation

```bibtex
@mastersthesis{saakstra2026harvar,
  author  = {Saakstra, Sake},
  title   = {{HAR-RS-DOW} Variance Forecasting for {Bitcoin}: Statistical
             Dominance and Applications in Risk Management and Option Pricing},
  school  = {Vrije Universiteit Amsterdam},
  year    = {2026},
  type    = {{MSc} thesis},
  note    = {{MSc} Econometrics \& Operations Research, Financial Track}
}
```

---

## Theoretical foundations

Full reference list in [`paper/paper.md`](paper/paper.md) §References. The paper builds on:

- **Corsi (2009)** — original HAR specification
- **Patton & Sheppard (2015)** — realized semivariance decomposition
- **Bollerslev, Patton & Quaedvlieg (2016)** — quarticity-adjusted HAR
- **Hansen (1994)** — autoregressive conditional density (skewed-t)
- **Acerbi & Szekely (2014)** — Expected Shortfall backtest (Z1 statistic)
- **Barndorff-Nielsen et al. (2008)** — realized kernel estimator
- **Basel Committee (2019)** — FRTB capital requirements

---

## License

MIT — see [LICENSE](LICENSE). Paper and code released for academic and educational use.

## Contact

Sake Saakstra — MSc EOR Financial Track, Vrije Universiteit Amsterdam
