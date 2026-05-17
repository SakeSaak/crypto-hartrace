# HAR-RS-DOW Variance Forecasting for BTC-EUR

> Econometric study of daily realised variance dynamics in cryptocurrency markets, establishing the Heterogeneous Autoregressive model with Realised Semivariance and Day-of-Week effects (HAR-RS-DOW) as the statistically dominant variance forecaster for BTC-EUR (with cross-asset replication on ETH-EUR and SOL-EUR, and outperformance over GARCH(1,1) benchmark), with canonical applications to risk management under Basel III and option pricing via Black-Scholes.

**MSc Econometrics & Operations Research thesis** — Financial Track, Vrije Universiteit Amsterdam (2025–2026)

- 📄 **Paper:** [`paper/paper.md`](paper/paper.md) | [`paper/paper.pdf`](paper/paper.pdf)
- 📊 **Empirical pipeline:** [`scripts/`](scripts/) (replication-ready)
- 🧪 **Tests:** [`tests/test_reproducibility.py`](tests/) — 20 passing tests, ~2s
- 📚 **References:** see paper §References

---

## Contribution

The paper makes three connected empirical contributions for cryptocurrency daily realised variance: primary results on BTC-EUR (n = 2 334 days, 3 April 2019 to 14 May 2026), with cross-asset replication on ETH-EUR (n = 2 081, since 7 February 2020) and SOL-EUR (n = 1 738, since 4 August 2021), and a non-HAR benchmark comparison against GARCH(1,1).

### 1. HAR-RS-DOW is the statistically dominant variance forecaster

We compare ten HAR-family specifications—HAR (Corsi 2009), HAR-Q (Bollerslev, Patton & Quaedvlieg 2016), HAR-RS (Patton & Sheppard 2015), and seven hybrid extensions—on in-sample Bayesian Information Criterion and on out-of-sample forecast performance. HAR-RS-DOW dominates all alternatives:

| Metric | HAR-RS-DOW | Next-best | Margin |
|---|---|---|---|
| In-sample BIC (n = 2 312) | −10 462 | −9 972 | ΔBIC = 490, Bayes factor ≈ 10¹⁰⁶ |
| OOS CRPS (n = 1 312) | **0.498** | 0.582 | 14% lower than next-best |
| OOS R² (log-RV) | **+0.439** | +0.246 | +78% relative |
| OOS QLIKE | **0.595** | 0.685 | 13% lower |
| Diebold-Mariano vs four nearest | dominant | — | p < 0.005 in all four pairwise tests |
| Multi-horizon h = 5 DM-statistic | **−5.46** | — | p < 10⁻⁷ |
| Hansen-Lunde-Nason MCS | **singleton {HAR-RS-DOW}** | — | at both 90% and 75% confidence, on CRPS and QLIKE |
| Cross-asset replication (ETH, SOL) | R² ∈ [+0.35, +0.50] | — | dominance preserved on all three crypto assets |
| Non-HAR benchmark vs GARCH(1,1) | margin +0.54 to +0.79 R² | GARCH < 0 | HAR exploits intraday info; GARCH worse than constant |

Estimated semivariance asymmetry $\hat\beta_d^- = 0.34$ versus $\hat\beta_d^+ = 0.18$ confirms the leverage effect operating at the variance level in BTC-EUR. Day-of-week dummies are statistically significant, with $\hat\gamma_{Mon} = +0.142$ and $\hat\gamma_{Sat} = -0.281$ capturing systematic weekly seasonality.

The Model Confidence Set procedure of Hansen, Lunde and Nason (2011), which corrects pairwise comparisons for multiple-testing across the full family of competitors, contains *only* HAR-RS-DOW at both 90% and 75% confidence levels on both CRPS and QLIKE losses. All four alternative HAR-family specifications—HAR-RS-Q-WE-X, HAR-RS-WE, HAR-RS-X, and HAR-WE—are formally eliminated as statistically inferior. This is the strongest possible statistical evidence of HAR-RS-DOW's unique forecasting superiority.

### 2. HAR-RS-DOW density forecasts are calibrated for Basel III Expected Shortfall

We assess the conditional density forecasts at three levels: tail-quantile coverage (Christoffersen 1998), Expected Shortfall magnitude (Acerbi-Szekely 2014), and full-density adequacy (Berkowitz 2001). Across this complete diagnostic battery the HAR-RS-DOW model proves adequate for regulatory risk-capital purposes.

**Full-density adequacy** — Berkowitz (2001) PIT test of the joint null $(\mu, \rho, \sigma_z) = (0, 0, 1)$:

| Density | LR-statistic | p-value | Verdict at 5% |
|---|---|---|---|
| Normal | 6.25 | 0.100 | **Pass** |
| Student-t (ν = 3) | 846.92 | < 0.001 | Reject (tails too heavy, $s_z = 0.50$) |
| Hansen skewed-t (ν̂ = 4.4, λ̂ = +0.01) | 7.22 | 0.065 | **Pass** |

**Expected Shortfall calibration** — Acerbi and Szekely (2014) Z1 backtests:

| Density specification | ES α=1% | ES α=5% | Z1 (α=1%) | Z1 (α=5%) | Calibration |
|---|---|---|---|---|---|
| Normal | −5.93% | −4.59% | **+2.11** | **+1.90** | Conservative |
| Student-t (ν=3) | −8.99% | −4.97% | **+1.73** | **+1.83** | Conservative |
| Hansen skewed-t (ν̂=4.4, λ̂=+0.01) | −7.86% | −5.06% | **+1.83** | **+1.81** | Conservative |
| Realised ES (1 312 days) | −5.84% | −3.91% | — | — | — |

Z1 > 0 across all specifications and at both significance levels indicates the model over-reserves capital relative to realised tail losses—the desired direction for regulatory adequacy under the Basel III Fundamental Review of the Trading Book (Basel Committee 2019).

### 3. HAR-σ̂ as input to Black-Scholes yields fair-value option pricing

Using HAR-RS-DOW's conditional volatility forecasts to price seven-day at-the-money BTC straddles produces the smallest systematic edge among three vol-input specifications:

| σ-input | Avg premium (€) | Avg realised payoff (€) | Avg edge (€) | Hit rate |
|---|---|---|---|---|
| Constant σ (full-sample mean) | 11.16 | 10.30 | +0.86 | 66% over-priced |
| Rolling 30-day σ | 10.59 | 10.30 | +0.29 | 63% |
| **HAR-RS-DOW σ̂** | **10.50** | 10.30 | **+0.20** | **60%** |

Across 1 305 OOS valuations, HAR-driven pricing produces the smallest average edge, consistent with fair-value market-making.

---

## Methodology

The paper develops the analysis in ten formal sections:

1. **Introduction** — motivation, literature placement, contributions, roadmap
2. **Literature** — HAR family and density-forecasting traditions
3. **Methodology** — HAR-RS-DOW specification, conditional density (Hansen 1994 skewed-t), evaluation criteria (CRPS, QLIKE, Diebold-Mariano)
4. **Data and realised-variance estimator** — Bitvavo 5-minute candles, Barndorff-Nielsen-Hansen-Lunde-Shephard (2008) realised kernel, stylised facts
5. **In-sample estimation results** — Table 1: ten-model BIC ranking with parameter estimates and HAC standard errors
6. **Out-of-sample forecast evaluation** — Tables 2-4: walk-forward CRPS / QLIKE / DM-tests at h ∈ {1, 5}; Hansen-Lunde-Nason (2011) Model Confidence Set with stationary block bootstrap
7. **Application: Value-at-Risk and Expected Shortfall** — Tables 5-7: Christoffersen (1998) coverage tests; Acerbi-Szekely (2014) Z1 backtests under three densities; Berkowitz (2001) PIT-test for full-density adequacy
8. **Application: Black-Scholes option pricing** — Table 8: edge analysis across three vol-input specifications
9. **Robustness** — cross-asset replication (ETH-EUR), sub-sample stability, density-specification robustness
10. **Conclusion** — limitations, implications, future work

---

## Repository structure

```
crypto_hartrace/
├── paper/                      The working paper
│   ├── paper.md                Source (Markdown with embedded LaTeX math)
│   ├── paper.pdf               Compiled via pandoc/xelatex
│   └── _archive_pre_academic.md   Pre-restructure draft (for diff)
│
├── src/hartrace/               Econometric library
│   ├── distributions.py        Hansen 1994 skewed-t implementation
│   ├── estimation.py           HAR-family OLS + MLE estimation
│   ├── features_v2.py          Daily realised measures (RV, RK, BPV, RS+, RS-, jump tests)
│   └── external_data.py        Optional macro covariates (FRED, yfinance)
│
├── scripts/                    Empirical pipeline (paper replication)
│   ├── A1_fetch_bitvavo_candles.py    OHLC data acquisition
│   ├── B1_signature_plot.py           Microstructure-noise analysis
│   ├── B2_recompute_realized_v2.py    Daily realised kernel computation
│   ├── C1_stylized_facts.py           Section 4.3 stylised facts
│   ├── D1_horse_race.py               Section 5 in-sample model comparison
│   ├── E1_walkforward.py              Section 6 OOS walk-forward
│   ├── E2_var_coverage.py             Section 7.1 VaR coverage diagnostics
│   ├── E4b_multihorizon.py            Section 6.3 multi-horizon results
│   ├── E3_mcs_test.py                 Section 6.4 Model Confidence Set
│   ├── E2b_christoffersen_var.py      Section 7.1 VaR coverage tests
│   ├── H4_expected_shortfall.py       Section 7.3 Acerbi-Szekely Z1
│   ├── E5_berkowitz_pit.py            Section 7.4 Berkowitz PIT test
│   ├── H5_density_robustness.py       Section 9.3 density-spec robustness
│   └── H6_option_pricing.py           Section 8 Black-Scholes valuation
│
├── tests/test_reproducibility.py   20-test reproducibility suite (~2s)
├── outputs/
│   ├── tables/                 Empirical tables (CSV/parquet)
│   ├── figures/                Plots (PNG)
│   └── latex/                  Overleaf-ready LaTeX explainer
└── data/                       Excluded from git (~186 MB); regenerable from Bitvavo REST API
```

---

## Reproducibility

```bash
pip install -r requirements.txt
pytest tests/test_reproducibility.py -v        # 20 tests, ~2s

# Re-run the empirical pipeline end-to-end
python scripts/A1_fetch_bitvavo_candles.py     # Acquire OHLC (~186 MB)
python scripts/B2_recompute_realized_v2.py     # Compute daily realised kernel
python scripts/D1_horse_race.py                # Table 1
python scripts/E1_walkforward.py               # Table 2
python scripts/E4b_multihorizon.py             # Table 3
python scripts/E3_mcs_test.py                  # Table 4 (Model Confidence Set)
python scripts/E2b_christoffersen_var.py       # Table 5 (VaR coverage tests)
python scripts/H4_expected_shortfall.py        # Table 6 (Expected Shortfall Z1)
python scripts/E5_berkowitz_pit.py             # Table 7 (Berkowitz PIT test)
python scripts/H5_density_robustness.py        # Robustness §9.3
python scripts/H6_option_pricing.py            # Table 8 (Black-Scholes pricing)
```

All outputs land in `outputs/tables/` (CSV / parquet) and `outputs/figures/` (PNG).

---

## Citation

```bibtex
@mastersthesis{saakstra2026harvar,
  author  = {Saakstra, Sake},
  title   = {{HAR-RS-DOW} Variance Forecasting for {BTC-EUR}: Statistical
             Dominance and Applications in Risk Management and Option Pricing},
  school  = {Vrije Universiteit Amsterdam},
  year    = {2026},
  type    = {Paper},
  note    = {{MSc} Econometrics \& Operations Research, Financial Track}
}
```

---

## References

Full reference list in [`paper/paper.md`](paper/paper.md) §References. Key methodological foundations:

- **Andersen, Bollerslev, Diebold & Labys (2003, Econometrica)** — realised volatility forecasting foundation
- **Bollerslev (1986, J. Econometrics)** — GARCH model
- **Corsi (2009, JFEcm)** — original HAR specification
- **Patton & Sheppard (2015, REStat)** — realised semivariance decomposition
- **Bollerslev, Patton & Quaedvlieg (2016, J. Econometrics)** — quarticity-adjusted HAR
- **Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, Econometrica)** — realised kernel
- **Hansen (1994, IER)** — autoregressive conditional density (skewed-t)
- **Hansen, Lunde & Nason (2011, Econometrica)** — Model Confidence Set test
- **Diebold & Mariano (1995, JBES)** — predictive accuracy comparison
- **Patton (2011, J. Econometrics)** — robust loss functions for noisy proxies
- **Christoffersen (1998, IER)** — VaR coverage tests
- **Berkowitz (2001, JBES)** — density forecast test via PIT
- **Acerbi & Szekely (2014, Risk)** — Expected Shortfall backtesting
- **Basel Committee on Banking Supervision (2019)** — FRTB capital framework

---

## License

MIT — see [LICENSE](LICENSE). Paper and code released for academic and educational use.

## Contact

Sake Saakstra — MSc Econometrics & Operations Research, Financial Track, Vrije Universiteit Amsterdam
