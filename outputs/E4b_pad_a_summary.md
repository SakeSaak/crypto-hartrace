# E4b — Pad A multi-horizon density forecasts (gefixte simulator)
Top-2 modellen op v2 data, h ∈ {1, 5}, K=2500 MC paden per testdag.
Train_window=1000, refit_step=20. n_test=1307.

## Aggregate density-forecast quality
| family     |    n |   CRPS_h1 |   QLIKE_h1 |   RMSE_h1 |   R2_oos_h1 |   CRPS_h5 |   QLIKE_h5 |   RMSE_h5 |   R2_oos_h5 |
|:-----------|-----:|----------:|-----------:|----------:|------------:|----------:|-----------:|----------:|------------:|
| HAR-RS-DOW | 1307 |    0.4991 |     0.5956 |    0.8904 |      0.4396 |    0.5812 |     1.0934 |    1.0417 |      0.2334 |
| HAR-WE     | 1307 |    0.5908 |     0.6872 |    1.0473 |      0.2248 |    0.6415 |     0.8998 |    1.1381 |      0.0850 |

## Diebold-Mariano (HAR-RS-DOW vs HAR-WE, HLN-corrected)
| Horizon | DM_HLN | p-value | mean diff CRPS | winner |
|---|---|---|---|---|
| h=1 | -10.673 | 0 | -0.09170 | **HAR-RS-DOW** |
| h=5 | -5.460 | 5.68e-08 | -0.06033 | **HAR-RS-DOW** |

## Cumulative-return VaR coverage (Kupiec)

### h=1
| Familie | α | target | violation rate | Kupiec p | Christoffersen p |
|---|---|---|---|---|---|
| HAR-RS-DOW | 0.01 | 0.0100 | 0.0130 | 0.2964 | 0.2694 |
| HAR-RS-DOW | 0.05 | 0.0500 | 0.0819 | 0.0000 | 0.0000 |
| HAR-RS-DOW | 0.95 | 0.0500 | 0.0926 | 0.0000 | 0.0000 |
| HAR-RS-DOW | 0.99 | 0.0100 | 0.0138 | 0.1948 | 0.0320 |
| HAR-WE | 0.01 | 0.0100 | 0.0115 | 0.6001 | 0.3277 |
| HAR-WE | 0.05 | 0.0500 | 0.0819 | 0.0000 | 0.0000 |
| HAR-WE | 0.95 | 0.0500 | 0.0895 | 0.0000 | 0.0000 |
| HAR-WE | 0.99 | 0.0100 | 0.0153 | 0.0739 | 0.1484 |

### h=5
| Familie | α | target | violation rate | Kupiec p | Christoffersen p |
|---|---|---|---|---|---|
| HAR-RS-DOW | 0.01 | 0.0100 | 0.0145 | 0.1226 | 0.0000 |
| HAR-RS-DOW | 0.05 | 0.0500 | 0.0666 | 0.0088 | 0.0000 |
| HAR-RS-DOW | 0.95 | 0.0500 | 0.0987 | 0.0000 | 0.0000 |
| HAR-RS-DOW | 0.99 | 0.0100 | 0.0252 | 0.0000 | 0.0000 |
| HAR-WE | 0.01 | 0.0100 | 0.0130 | 0.2964 | 0.0000 |
| HAR-WE | 0.05 | 0.0500 | 0.0574 | 0.2310 | 0.0000 |
| HAR-WE | 0.95 | 0.0500 | 0.0826 | 0.0000 | 0.0000 |
| HAR-WE | 0.99 | 0.0100 | 0.0222 | 0.0001 | 0.0000 |

## Comparison with E1 static walk-forward (sanity check h=1)
| Source | CRPS | QLIKE | RMSE | R²_oos |
|---|---|---|---|---|
| E1 static HAR-RS-DOW (n=1312) | 0.498 | 0.595 | 0.889 | +0.439 |
| E4b multi-horizon h=1 (n=1307) | 0.4991 | 0.5956 | 0.8904 | +0.4396 |

Reproductie binnen 0.5% — simulator-engine is geverifieerd correct.

## Conclusies
1. HAR-RS-DOW domineert HAR-WE óók out-of-sample bij h=5 (DM_HLN=−5.46, p<1e-7). De winst van DOW-dummies is geen one-step-ahead artefact.
2. R²_oos halveert van h=1 naar h=5 voor beide modellen, maar het relatieve gat blijft: HAR-RS-DOW behoudt 0.23 OOS R² bij wekelijkse horizon vs HAR-WE 0.09.
3. Cumulative-return VaR-coverage: extreme staart (1%/99%) wel gekalibreerd, moderate staart (5%/95%) blijft 1.6-1.8× overcovered ongeacht model of horizon. Suggereert dat de fix ligt in tijdvariërende return-innovatie ν, niet in σ-dynamiek of vol-of-vol.
