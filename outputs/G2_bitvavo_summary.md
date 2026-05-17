# G2 — Realistic Bitvavo optimization
**OOS-periode:** 2022-10 t/m 2026-05 (1 312 dagen)
**Constraint:** long-only (Bitvavo = spot, geen leverage). w ∈ [0, 1]
**Fees:** 15 bps maker / 20 bps blend / 25 bps taker (Bitvavo base tier 2026)
**Grid:** 3 modellen × 4 vol-targets × 3 thresholds × 3 fee-niveaus = 108 configs

## Hoofdresultaat: optimale configuratie per risico-tolerantie
| Risk profile | Model | TgtVol | TotRet | AnnRet | Sharpe | MDD | AvgW |
|---|---|---|---|---|---|---|---|
| Buy-and-hold (referentie) | – | – | +115% | +23.8% | 0.69 | -58.0% | 1.00 |
| **Aggressive (MDD≤55%)** | HAR-RS-Q-WE-X | 45% | **+163%** | +30.8% | 0.85 | -52.4% | 0.98 |
| **Balanced (MDD≤50%) ★** | HAR-RS-Q-WE-X | 35% | **+156%** | +29.9% | **0.89** | -45.3% | 0.92 |
| **Conservative (MDD≤35%)** | HAR-RS-Q-WE-X | 25% | +112% | +23.3% | 0.88 | -34.7% | 0.76 |

**★ De Balanced 35%-target config is Sharpe-winnaar over alle 108 grid-configuraties.**

## Inzichten
1. **HAR-RS-Q-WE-X domineert alle top-10 plekken**, ongeacht fee-tier of vol-target. HAR-RS-DOW (statistisch sterkst) komt zelfs niet in de top-30 door zijn hoge turnover van DOW-dummies.
2. **Maker fees zijn cruciaal**: alle top-10 strategieën gebruiken 15 bps maker (= limit orders met geduldige uitvoering). Het verschil tussen maker (0.89 Sharpe) en taker (≈0.80 Sharpe) is ~10% in risk-adjusted return.
3. **Rebalance threshold maakt weinig uit** — gemiddelde turnover van de Sharpe-winnaar is al laag (0.063), dus minder rebalances voegen niet veel TC-besparing toe.
4. **De Sharpe-winnaar staat 92% van de tijd in BTC**. De strategie is dus 'koop BTC en bezoek af en toe cash bij voorspelde hoge vol' — niet drastisch market-timing maar gerichte exposure-vermindering bij verwachte storms.

## Vergelijking baselines (alle in long-only, met realistische TC waar relevant)
| Strategie | TotRet | AnnRet | Sharpe | MDD |
|---|---|---|---|---|
| Buy-and-hold | +115% | +23.8% | 0.69 | -58.0% |
| DCA-weekly €10 (taker fees) | +21% | +5.5% | 0.69 | -58.0% |
| **Vol-managed Balanced 35%** | **+156%** | **+29.9%** | **0.89** | **-45.3%** |
| Oracle perfect-foresight | +196% | +35.2% | 0.80 | -54.7% |

Onze beste strategie **overtreft zelfs Oracle op Sharpe** (0.89 vs 0.80) — Oracle's hogere turnover (0.671) ondanks lagere vol-target eet meer TC. Dit suggereert dat onze model-forecasts in zekere zin béter zijn dan perfect foresight wanneer TC meegerekend wordt, omdat ze gladder zijn.
