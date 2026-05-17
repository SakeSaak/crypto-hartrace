# Changelog

All notable changes to the paper and supporting econometric pipeline.

## [2026-05-17] Repository repositioned as pure econometric study

Removed all trading-bot related content from the public repository to align with
the academic focus of the paper. The trading bot is engineering infrastructure
operated by the author privately and is not part of the paper's contribution.

Removed from public tracking:
- All live-trading source code (`src/hartrace/live/`)
- All trading-strategy R&D scripts (H7 through H11)
- Trading bot entry point and operational tooling
- Bot launchd configurations and operator runbook

The remaining content is organised around the paper's three contributions:
HAR-RS-DOW as the statistically dominant variance forecaster, Basel III
Expected Shortfall adequacy, and Black-Scholes option pricing.

## [2026-05-17] Academic paper structure created (paper/paper.md)

Restructured the working paper following conventions observed in canonical
HAR-family publications (Corsi 2009, JFE; Patton-Sheppard 2015, REStat;
Bollerslev-Patton-Quaedvlieg 2016, J. Econometrics):

Structure: Abstract → 1. Introduction → 2. Literature → 3. Methodology →
4. Data and realised-variance estimator → 5. In-sample estimation results →
6. Out-of-sample forecast evaluation → 7. Application: VaR + ES →
8. Application: Black-Scholes option pricing → 9. Robustness → 10. Conclusion
→ References.

## [2026-05-15] Density-robustness analysis added

Expected Shortfall backtesting extended to three density specifications:
Normal, Student-t(ν=3), and Hansen (1994) skewed-t with empirically estimated
parameters. Acerbi-Szekely Z1 statistic positive (conservative) under all
three specifications and at both significance levels (α ∈ {1%, 5%}).

## [2026-05-14] Option-pricing application added

Black-Scholes valuation of 7-day at-the-money BTC straddles using three
volatility-input specifications: constant σ (sample mean), rolling 30-day
realised σ, and HAR-RS-DOW conditional forecast σ̂. HAR-driven pricing
produces fair-value closest to market-clearing (edge €0.20 vs €0.86 and €0.29).

## [2026-05-13] Out-of-sample walk-forward evaluation completed

1 312-day OOS window evaluation under walk-forward re-estimation every 20 days.
HAR-RS-DOW dominates all alternative HAR-family specifications on CRPS (0.498
vs 0.582 for the next-best), QLIKE, RMSE, and R² OOS. Diebold-Mariano test
rejects equal predictive accuracy at p < 0.005 against all four nearest
competitors.

## [2026-05-12] In-sample horse race completed

Ten HAR-family specifications estimated on 2 312 in-sample days by OLS with
Newey-West HAC standard errors. HAR-RS-DOW achieves BIC of −10 462, with
ΔBIC of 490 over the next-best specification (Bayes factor ≈ 10^106).

## [2026-05-10] Realised-variance pipeline established

Daily realised variance computed from Bitvavo BTC-EUR 5-minute candles using
the Barndorff-Nielsen, Hansen, Lunde & Shephard (2008) realised kernel with
Parzen weighting. Five-minute frequency selected on basis of signature plot
analysis showing acceptable microstructure-noise levels.

## [2026-05-08] Repository initialised

MSc Econometrics & Operations Research thesis project, Financial Track,
Vrije Universiteit Amsterdam (academic year 2025-2026).
