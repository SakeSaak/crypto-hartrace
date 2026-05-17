# Stylized facts (1–3) — BTC-EUR

Sample: 2025-03-19 → 2026-02-01 (316 days, 15-min RV measures)

## Fact 1 — Distribution

- Daily returns: mean=-0.00059, std=0.0206, skew=-0.027, **excess kurt = +2.455**, JB p=3.65e-17
- log RV    : mean=-8.057, std=0.881, skew=-0.308, **excess kurt = +0.338**, JB p=4.41e-02
- Hansen skew-t fit on returns: **ν̂ = 3.77**, **λ̂ = -0.066**, log-lik = -427.74
- Hansen skew-t fit on log RV : ν̂ = 13.09, λ̂ = -0.179
- GPD shape ξ̂ (95-pct excess of |r_d|): **+nan**

## Fact 2 — Volatility clustering

Ljung-Box and ARCH-LM (selected lags):

| series   | test      |   lag |   stat |   pvalue |
|:---------|:----------|------:|-------:|---------:|
| r_d      | Ljung-Box |     5 |   2.31 | 0.805    |
| r_d      | Ljung-Box |    10 |   6.66 | 0.757    |
| r_d      | Ljung-Box |    22 |  22.6  | 0.426    |
| |r_d|    | Ljung-Box |     5 |   9.54 | 0.0893   |
| |r_d|    | Ljung-Box |    10 |  12.4  | 0.262    |
| |r_d|    | Ljung-Box |    22 |  23.8  | 0.359    |
| r_d^2    | Ljung-Box |     5 |   9.98 | 0.0757   |
| r_d^2    | Ljung-Box |    10 |  11.1  | 0.348    |
| r_d^2    | Ljung-Box |    22 |  30.2  | 0.113    |
| log_RV   | Ljung-Box |     5 |  89.6  | 8.02e-18 |
| log_RV   | Ljung-Box |    10 | 221    | 7.95e-42 |
| log_RV   | Ljung-Box |    22 | 307    | 5.82e-52 |
| r_d      | ARCH-LM   |     5 |   8.62 | 0.125    |
| r_d      | ARCH-LM   |    10 |  10.1  | 0.436    |
| r_d      | ARCH-LM   |    22 |  29.9  | 0.122    |

## Fact 3 — Long memory

- R/S Hurst on |r_d| : H = +0.710 (SE 0.014)
- R/S Hurst on log RV: H = +0.875 (SE 0.016)
- GPH d on |r_d|     : d = +0.268 (SE 0.156, p=0.0845)
- GPH d on log RV    : d = +0.453 (SE 0.156, p=0.00358)

Figures saved in `outputs/figures/`, tables in `outputs/tables/`.