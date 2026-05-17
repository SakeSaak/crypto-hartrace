# Stylized facts v2 — BTC-EUR, 5-min RK, n≈2300

Sample: 2019-04-03 → 2026-05-15 (2334 days)

## Fact 1 — Distribution (v2)

- daily returns r_d: skew=-1.329, excess kurt=+21.585, skew-t ν̂=3.02, λ̂=+0.014
- log RK         : skew=-0.121, excess kurt=+0.356, skew-t ν̂=16.81, λ̂=-0.069
- log RV (simple): skew=+0.416, excess kurt=+0.966
- GPD ξ̂ (|r| > q95): +0.133 (positive ⇒ heavy-tailed regime)

## Fact 2 — Clustering (v2)

|                    |   5 |   10 |   22 |
|:-------------------|----:|-----:|-----:|
| ('log_RK', 'LB')   |   0 |    0 |    0 |
| ('r_d', 'ARCH-LM') |   0 |    0 |    0 |
| ('r_d', 'LB')      |   0 |    0 |    0 |
| ('r_d^2', 'LB')    |   0 |    0 |    0 |
| ('|r_d|', 'LB')    |   0 |    0 |    0 |

## Fact 3 — Long memory + roughness

- R/S Hurst H(|r_d|)  = +0.727 (SE 0.012)
- R/S Hurst H(log RK) = +0.890 (SE 0.012)
- GPH d (log RK)      = +0.653 (SE 0.093, p=1.718e-12)
- Hurst log-RK *increments* (Takaishi): H(q=0.5)=0.063, H(q=1.0)=0.063, H(q=2.0)=0.064
  (H_incr < 0.5 ⇒ rough volatility, à la Gatheral-Jaisson-Rosenbaum 2018)

## Fact 4 — Jump dynamics

- jump rate (BNS α=0.001): 11.70%
- on jump-days, J/RV: mean = 0.253
- AR(1) on jump indicator: ρ̂_1 = +0.104  (weak clustering if non-zero)

## Fact 5 — Inverse-leverage test (Bouri-style)

Regression: log RK_{t+1} on log RK_t, r_t, 1{r_t<0}·|r_t|, 1{r_t>0}·|r_t|. HAC SE.

| param      |    beta |   se_hac |      p |
|:-----------|--------:|---------:|-------:|
| const      | -3.6666 |   0.2813 | 0.0000 |
| log_RV_lag |  0.5283 |   0.0333 | 0.0000 |
| r_lag      | -0.7648 |   0.4146 | 0.0651 |
| neg_abs    |  3.4144 |   0.9569 | 0.0004 |
| pos_abs    |  2.6495 |   1.0916 | 0.0152 |

  R² = 0.3238, n = 2333

  Standard leverage pattern: β_neg (+3.414) > β_pos (+2.650).

## Fact 6 — Periodicity

Day-of-week mean log-RK:
  - Mon: -7.266
  - Tue: -7.404
  - Wed: -7.340
  - Thu: -7.427
  - Fri: -7.462
  - Sat: -8.558
  - Sun: -8.034

- F-test (all DOW means equal): F = 51.983, p = 0.0000
- Weekend dummy effect (HAC): β = -0.9184, SE = 0.0618, p = 0.0000