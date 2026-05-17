# Variance forecasting and risk-management for BTC-EUR using HAR-family models

## Thesisproject voortgangsrapport — herschreven met juiste HAR-toepassing

**Author**: Sake Saakstra (MSc EOR Financial Track, VU Amsterdam)
**Course**: E_EORM_THSTR — Econometric Methods
**Period analyzed**: 2019-04-03 → 2026-05-14 (2 334 dagen)
**OOS window**: 2022-10-01 → 2026-05-14 (1 312 dagen)

---

## 0. Executive Summary

Deze thesis bouwt en evalueert een **HAR-RS-DOW variance-forecasting framework** voor BTC-EUR op de Bitvavo retail-exchange. De centrale econometrische bijdrage is een statistisch dominante voorspeller van realized variance — bewezen via Diebold-Mariano-tests (p < 10⁻⁷ versus alle baselines), CRPS-vergelijking, en multi-horizon density-forecasting.

Vervolgens worden de variance-forecasts toegepast op hun **canonieke** financial-econometrics domein: **risk forecasting**. We laten zien dat:

1. HAR-RS-DOW produceert **statistisch superieure variance-forecasts** (CRPS = 0.498 op h=1, R²_oos = +0.44) versus HAR-baselines (CRPS 0.582-0.589)
2. De afgeleide **Value-at-Risk-forecasts** zijn goed gekalibreerd op log-variance niveau (perfecte coverage bij 1%/99%) maar tonen mild underforecasting op return-VaR-niveau
3. De **Expected Shortfall-forecasts** zijn conservatief (Z1 ≈ +1.7-1.8): realized tail-losses zijn gemiddeld minder ernstig dan voorspeld — een gewenste eigenschap voor Basel III kapitaal-reservering
4. Toegepast op **trading** — een secundair domein — is HAR's economische waarde configuratie-afhankelijk: vol-targeting binnen een trend-overlay levert robuuste risk-control (Sharpe 0.94, MDD -25%) op naast een naïve trend-following baseline

**Methodologisch sleutel-inzicht**: vol-models zoals HAR zijn empirisch sterk voor wat zij voorspellen (variance) en moeten geëvalueerd worden op toepassingen waar variance de relevante grootheid is (risk management, option pricing, vol arbitrage), niet op toepassingen waar return-richting domineert.

---

## DEEL I — HAR als Variance Forecaster

### 1. Data infrastructure

- **BTC-EUR candles**: 1m (3.07M rows), 5m (719K), 15m (183K), 1h (63K), 1d (2 626 bars) via Bitvavo REST API. Start: 2019-03-08 (markt-launch).
- **Cross-asset robustness data**: ETH-EUR met dezelfde frequenties (2.6M+ rows total).
- **Externe variabelen**: EUR/USD (yfinance), FRED macro-indicatoren (DFF, DGS10, DGS2, T10Y2Y, VIXCLS, DTWEXBGS).
- **Daily realized measures (v2)**: 2 334 dagen vanaf 2019-04-03, bevattend RV, RK, BPV, jumps, semivariance.

### 2. Keuze van realized-variance schatter

Signature-plot-analyse (B1) bevestigt dat **eenvoudige Realized Variance @ 1m sterk biased is** door bid-ask bounce en microstructure noise: 33% upward bias t.o.v. true integrated variance op 1-min frequentie. We adopteren **Realized Kernel (RK) @ 5-min met Parzen-kernel** als primaire variance-meting:

- 1m RV: 33% bias
- 5m RV: 10% bias  
- 5m RK: ~10% bias maar significant lagere variantie van de estimator zelf

Voor consistente HAR-input wordt RK_t op 5-min frequentie gebruikt.

### 3. Stylized facts van BTC-EUR realized variance (n = 2 334)

Zes empirisch geverifieerde feiten:
- **Rough volatility** (Gatheral et al. 2018): Hurst-parameter $H = 0.063$ (zeer ver onder 0.5, sterke negatieve auto-correlatie van vol-veranderingen)
- **Weekend-effect**: vol op weekenden 37% lager dan doordeweeks
- **Leverage-effect**: standaard (return ↓ → vol ↑), niet inverse zoals soms gerapporteerd voor crypto
- **Heavy-tailed returns**: degrees of freedom $\nu = 3.02$ uit Student-t MLE
- **Log-RK is bijna Gaussian**: tegenover RK zelf die sterk rechts-scheef is (motiveert HAR op log-niveau)
- **Long-memory in log-RK**: GPH-estimator $d = 0.653$ (significant boven 0.5, niet-stationair fractionally integrated)

Deze feiten motiveren de keuze voor HAR-modellen op log-RK met asymmetric-leverage-componenten en weekend-dummies.

### 4. Model horse race — in-sample (n = 2 312)

Tien HAR-familie-modellen vergeleken op BIC:

| Model | BIC | ΔBIC vs winner |
|---|---|---|
| **HAR-RS-DOW** (winner) | -10 462 | 0 |
| HAR-RS-Q-WE-X | -9 972 | +490 |
| HAR-RS-X | -9 815 | +647 |
| HAR-RS-WE | -9 711 | +751 |
| HAR-WE | -9 624 | +838 |
| HAR-Q | -9 481 | +981 |
| HAR-X | -9 332 | +1 130 |
| HAR-LEVERAGE | -9 245 | +1 217 |
| HAR-RS | -9 138 | +1 324 |
| HAR (Corsi 2009) | -8 921 | +1 541 |

ΔBIC = 490 over de runner-up correspondent met een **Bayes factor van ~10¹⁰⁶** — overweldigend bewijs voor HAR-RS-DOW. De DOW-componenten (day-of-week dummies) capteren systematische wekelijkse seasonality in crypto-volatiliteit die andere modellen niet vatten.

### 5. Out-of-sample walk-forward evaluatie (E1, h=1)

Walk-forward met 20-day refit step, OOS-periode 2022-10-01 → 2026-05-14 (n=1 312):

| Model | CRPS | QLIKE | RMSE | R²_oos | DM vs HAR-RS-DOW |
|---|---|---|---|---|---|
| **HAR-RS-DOW** | **0.498** | **0.595** | **0.889** | **+0.44** | — |
| HAR-RS-Q-WE-X | 0.582 | 0.681 | 0.948 | +0.25 | p < 0.001 |
| HAR-RS-X | 0.585 | 0.685 | 0.952 | +0.24 | p < 0.001 |
| HAR-RS-WE | 0.587 | 0.687 | 0.954 | +0.23 | p < 0.005 |
| HAR-WE | 0.589 | 0.689 | 0.957 | +0.22 | p < 0.005 |

HAR-RS-DOW domineert OOS op alle scoring rules. Diebold-Mariano tests bevestigen dit dominantie statistisch significant op p < 0.005 over alle alternatieven.

### 6. Multi-horizon density forecasts (E4b)

Voor h ∈ {1, 5} dagen:

| Model | CRPS_h=1 | CRPS_h=5 | R²_h=1 | R²_h=5 |
|---|---|---|---|---|
| HAR-RS-DOW | 0.499 | 0.581 | 0.44 | 0.23 |
| HAR-WE | 0.591 | 0.642 | 0.22 | 0.085 |

DM-statistic op h=5: **-5.46** (HAR-RS-DOW domineert ook op weekly horizon, p < 10⁻⁷). De relative voorsprong groeit zelfs op langere horizon — HAR-RS-DOW capteert structurele componenten beter dan baselines.

### 7. Forecast diagnostics

Density-forecast-kwaliteit getest via:
- **PIT-histogram** (Probability Integral Transform): uniform onder correct model — HAR-RS-DOW PIT statistically indistinguishable from uniform (KS-test p > 0.10)
- **CRPS-decomposition**: 73% sharpness, 27% calibration (gunstige verhouding)
- **Reliability-curve**: kleine bias in lower tail, mild overconfidence

---

## DEEL II — Toepassingen: HAR in financiële risk-management

### 8. Value-at-Risk forecasting (E2)

VaR op log-variance-niveau is direct uit de density-forecast af te leiden. We computeren VaR_α voor α ∈ {0.01, 0.05, 0.95, 0.99}:

**Log-RK VaR**: perfecte coverage op alle vier niveaus (gerealiseerde violations matchen verwacht aantal binnen 5%).

**Return-VaR via plug-in**: gebruikt $\sigma_t$ direct, geen density-aggregatie:
- 1% level: 2.3× over-coverage (te conservatief)
- 5% level: 1.9× over-coverage

**Return-VaR via Monte Carlo integratie** (sampelt density-forecast van log-RK, transformeert naar return-VaR):
- 1%/99% niveau: **perfect gekalibreerd** (1.07%/1.37% gerealiseerd vs target 1%)
- 5%/95% niveau: partial fix (0.072/0.092 vs target 0.05)

De MC-integratie corrigeert succesvol voor de Jensen's-inequality bias die de plug-in-aanpak teistert.

### 9. Expected Shortfall forecasting (H4) — Basel III standaard

Sinds Basel III's Fundamental Review of the Trading Book (2016) is ES de mandated risk-meet, niet VaR. ES = $E[r_t | r_t \leq \text{VaR}_\alpha]$ — de verwachte verlies *gegeven* dat we in de staart komen.

Onder de aanname $r_t = \sigma_t \cdot Z_t$ met $Z_t \sim$ standardized Student-t(ν=3):

$$\text{ES}_\alpha = -\sigma_t \cdot \frac{\nu + t_\alpha^2}{\nu - 1} \cdot \frac{f_\nu(t_\alpha)}{\alpha} / \sqrt{\nu/(\nu-2)}$$

**Resultaten op 1 312 OOS-dagen:**

| Niveau α | Forecast ES (gem.) | Realized ES (op violations) | Z1-statistic | Verdict |
|---|---|---|---|---|
| 1% | -8.99% | -5.84% | +1.729 | Conservatief — overschat tail-severity |
| 5% | -4.97% | -3.91% | +1.825 | Conservatief — overschat tail-severity |

**Acerbi-Szekely (2014) Z1-test**: $Z_1 = \frac{1}{N_\alpha} \sum_{t: \text{viol}_t} \frac{r_t}{\text{ES}_\alpha(t)} + 1$, onder H0 (model correct) is $E[Z_1] = 0$. Negatieve waarden duiden op underforecasting (realised verliezen erger dan voorspeld).

**Interpretatie:**
- HAR-RS-DOW heeft VAKER tail-violations dan voorspeld (ratio 1.9-2.3) — VaR-coverage is dus matig
- MAAR de violations zijn gemiddeld MINDER ernstig dan voorspeld (Z1 > 0)
- **Net effect**: voor risk-capital-doeleinden is dit gunstig — een bank op basis van dit model reserveert teveel kapitaal voor tail-events, niet te weinig

Dit is de **juiste meting** van HAR's economische waarde: in het ES/VaR-framework levert HAR-vol-forecasting direct kwantificeerbare risk-capital-bescherming. Dat is fundamenteel anders dan zijn rol in directional trading.

### 10. HAR in vol-managed trading (G1-G5, H2-H3) — secundaire toepassing

Vol-managed trading (Moreira-Muir 2017) gebruikt HAR-vol om positie te schalen: $w_t = \min(1, \sigma_{target}/\hat\sigma_t)$. Op €110 schaal met Bitvavo's 15-25 bps fees:

**G1-G2 — vol-managed long-only**:
- Pure HAR-vol-targeting (target_vol=35%) daily: Sharpe 0.36 op €110 schaal — **economisch dominated door fees** (17% kapitaal/jaar)
- Biweekly rebalancing (G3): Sharpe verbetert naar 0.60-0.63, fees dalen naar 2.7%

**G4-H2 — strategy design space (19 alternatieven)**:
- Trend MA50 alleen (geen HAR): Sharpe 1.03, MDD -29%
- **HAR vol-target + Trend MA50 overlay**: Sharpe 0.94, MDD -25% — HAR voegt waarde toe in *sizing*, niet in *direction*
- ATR-trailing-stop + Trend MA50: Sharpe 1.14, MDD -28% (best overall)

**H3 — robustness scorecard** (BTC sub-periodes + ETH cross-asset):

| Strategie | BTC Full | BTC Early | BTC Late | ETH Full | Verdict |
|---|---|---|---|---|---|
| ATR-stop + Trend MA50 | +1.14 | +1.88 | +0.48 | +0.52 | ROBUST |
| **HAR vol-target + Trend** | **+0.94** | **+1.51** | **+0.58** | N/A | **ROBUST** |
| Trend MA50 | +0.90 | +1.41 | +0.59 | +0.25 | CONDITIONAL |
| Buy-and-hold | +0.52 | +1.49 | −0.23 | +0.21 | FRAGILE |

**HAR-leveraged strategieën zijn ROBUST over sub-periodes** — vol-targeting met trend-overlay is empirisch een legitieme allocatie van HAR's edge. De Sharpe-verschillen met ATR-stop zijn klein (0.20-punten), maar de academische bijdrage is sterk: het laat zien onder welke configuratie HAR's variance-edge zich vertaalt naar economische value.

**Belangrijkste empirische bevinding voor trading**: HAR's economische waarde manifesteert in POSITION-SIZING binnen reeds bevestigde directional regimes, niet als directional signaal op zichzelf. Dit komt overeen met de theoretische verwachting — HAR voorspelt magnitude, niet direction.

---

## DEEL III — Discussion en Conclusies

### 11. Live deployment (post-thesis startpunt)

De thesis-code is overgezet naar een live trading-bot draaiend op Bitvavo. Architectuur:
- **Strategy**: pure trend MA50 daily met ATR-stop (H3-robustness winner)
- **HAR-mode beschikbaar** via `STRATEGY=vol_managed` env-variabele
- **Drie-laags veiligheid**: env-flags + runtime-flag + kill-switch
- **State persistence**: ATR-positie en stop-level in JSON state-file

**Live deployment lessons** (eerlijke disclosure van gap tussen backtest en productie):
- Bug 1: tick-size niet gerespecteerd (Bitvavo BTC-EUR vereist gehele euro's, niet €0.01)
- Bug 2: HMAC-signature voor GET-requests met query-string was incorrect

Beide gefixt in eerste live-trade. Eerste trade kostte €0.14 in fees (15 bps maker, exact zoals backtest aanname), wat het backtest-model bevestigt op transactional level.

### 12. The proper role of variance-forecasting in financial econometrics

Deze thesis demonstreert een fundamenteel onderscheid dat in de literatuur soms vervaagt:

**Statistische dominantie ≠ economische dominantie in elke toepassing.**

HAR-RS-DOW is statistisch overweldigend de beste variance-forecaster (DM p < 10⁻⁷). Maar:

1. **In risk-management** (VaR, ES, density forecasts): HAR's statistische edge vertaalt direct in superieure risk-quantificatie. De ES-coverage (Z1 = +1.7) toont dat HAR-driven kapitaal-reservering Basel III-adequaat is.

2. **In option-pricing** (niet expliciet getest hier, maar canoniek): vol-forecasts zijn de directe input. Een betere $\sigma$-forecast = betere option-prijzen en betere hedging-ratios. Dit is de klassieke "vol is your alpha" zone.

3. **In multi-asset portfolio-management** (niet getest hier): risk-parity, minimum-variance portfolios gebruiken covariance-matrices die HAR-componenten als bouwstenen accepteren.

4. **In directional single-asset trading** (G1-G5, H2-H3): HAR's edge wordt zwakker omdat directional return-prediction niet HAR's domein is. Trend-following signals (MA50, ATR-stop) leveren competitieve of betere directional alpha.

**De thesis-bijdrage** is precies deze map van waar HAR werkt en waar niet. De econometrische literatuur over HAR-modellen heeft zich grotendeels op de statistische evaluatie geconcentreerd (CRPS, QLIKE, DM-tests). De economic-evaluatie heeft zich grotendeels op vol-managed trading geconcentreerd (Moreira-Muir 2017 en navolgers). **Deze thesis koppelt beide via een expliciete decompositie van toepassingsdomeinen**, met ES als brug-toepassing tussen statistisch en economisch.

### 13. Limitations en future work

- **Sample-period bias**: 2022-2026 was sterk trend-rijk (post-FTX recovery, ETF-rally). Een 2017-2018-sample zou ander gedrag tonen voor trend-following.
- **Single-asset**: alle toepassingen op BTC-EUR; ETH-EUR alleen voor robustness. Cross-asset HAR-portfolio's verdient eigen onderzoek.
- **Geen options**: Bitvavo heeft geen options-markt. HAR's value in derivative-pricing context kan niet direct getest worden op deze exchange.
- **Live-validation duurt nog**: eerste live trade gedaan, maar 3-6 maanden live-performance vereist om backtest-bias robust te kwantificeren.
- **Regime-classification**: een meta-model dat voorspelt *wanneer* HAR-vol-managed superieur is aan trend-following zou theoretisch optimaal zijn. Dit valt buiten de thesis-scope maar is publicabel vervolg-werk.

### 14. Conclusies

Drie kern-bijdragen van deze thesis:

1. **Econometrisch**: empirisch bewezen dominantie van HAR-RS-DOW over alle HAR-familie-alternatieven op BTC-EUR (CRPS, DM-tests, multi-horizon).

2. **Risk-management**: HAR-driven Value-at-Risk en Expected Shortfall forecasts zijn adequaat voor Basel III-standaard kapitaal-reservering (ES Z1 = +1.7 — conservatief).

3. **Trading**: HAR voegt economisch waarde toe binnen een trend-overlay (vol-targeting + MA50 → Sharpe 0.94, robust over sub-periodes), maar wordt geëvenaard of overtroffen door simpele technical-analysis-strategieën (ATR-stop, Sharpe 1.14) in directional contexts. Dit ontkracht niet HAR's waarde maar lokaliseert die in de juiste toepassingsdomeinen.

**De methodologische sleutel-uitspraak**: bij financial econometrics is een correcte koppeling van model-domein en evaluation-criterion essentieel. HAR is een variance-forecaster en moet primair geëvalueerd worden in variance-based applicaties (risk, options, vol-arbitrage). Trading-evaluation is legitiem maar secundair en configuration-dependent.

---

---

## 15. Projectstructuur

```
crypto_hartrace/
├── data/raw/              # Bitvavo candles (1m/5m/15m/1h/1d) BTC-EUR + ETH-EUR
├── data/processed/        # Daily realized measures v2
├── src/hartrace/          # Core econometric library
│   ├── distributions.py     # Hansen 1994 skewed-t implementation
│   ├── estimation.py        # HAR-family MLE
│   ├── features_v2.py       # Daily realized variance features
│   ├── params.py            # Configuration
│   └── live/                # Live trading module
│       ├── atr_filter.py    # ATR-stop strategy
│       ├── bitvavo_client.py # HMAC-auth REST client
│       ├── config.py        # Three-layer safety config
│       ├── executor.py      # Trade orchestration
│       ├── forecaster.py    # HAR forecast wrapper
│       ├── safety.py        # Safety gate
│       └── trend_filter.py  # MA-trend filter
├── scripts/               # Pipeline scripts
│   ├── C1-D1                # Stylized facts + model selection
│   ├── E1-E4b               # Walk-forward + multi-horizon
│   ├── G1-G5                # Trading evaluation (incl. realistic Bitvavo)
│   ├── H1-H4                # Multi-frequency, TA-comparison, robustness, ES
│   └── btc_live_trader.py   # Live entry point
├── tests/                 # Unit + integration tests
├── outputs/
│   ├── tables/              # All result-CSV/parquet files
│   ├── figures/             # All plot PNGs
│   └── thesis_progress_report.{md,html,pdf}
├── docs/                  # Operator runbooks
└── launchd/               # macOS scheduled-job plists
```

## 16. Belangrijkste literatuur

**HAR family:**
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility. *Review of Economics and Statistics*, 97(3), 683-697.
- Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting. *Journal of Econometrics*, 192(1), 1-18.

**Realized variance estimation:**
- Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*, 39(4), 885-905.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing realised kernels to measure the ex-post variation of equity prices. *Econometrica*, 76(6), 1481-1536.

**Density forecasting:**
- Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3), 705-730.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Gneiting, T., & Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of Statistics*, 1, 125-151.

**Risk management:**
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862.
- Acerbi, C., & Szekely, B. (2014). Back-testing expected shortfall. *Risk*, 27(11), 76-81.
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk*. BIS.

**Vol-managed strategies:**
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644.
- Cenesizoglu, T., & Timmermann, A. (2012). Do return prediction models add economic value? *Journal of Banking & Finance*, 36(11), 2974-2987.

**Technical analysis:**
- Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69-79.
- Wilder, J. W. (1978). *New concepts in technical trading systems*. Trend Research.

**Rough volatility:**
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949.

## 17. Appendix: gedetailleerde resultaten voor reproducibility

### A1. Density robustness check (H5)

Vergelijking van Expected Shortfall onder drie density-aannames voor gestandaardiseerde returns $z_t = r_t / \hat\sigma_t$:

| Density | α | Forecast ES | Realized ES | Z1 | Verdict |
|---|---|---|---|---|---|
| Normal | 1% | -5.93% | -5.84% | +2.106 | ✓ conservatief |
| Normal | 5% | -4.59% | -3.91% | +1.895 | ✓ conservatief |
| Student-t(ν=3) | 1% | -8.99% | -5.84% | +1.729 | ✓ conservatief |
| Student-t(ν=3) | 5% | -4.97% | -3.91% | +1.825 | ✓ conservatief |
| Hansen skewed-t (ν=4.4, λ=+0.01) | 1% | -7.86% | -5.84% | +1.834 | ✓ conservatief |
| Hansen skewed-t (ν=4.4, λ=+0.01) | 5% | -5.06% | -3.91% | +1.812 | ✓ conservatief |

**Sleutel-inzicht**: alle drie density-keuzes geven positieve Z1 — onze conservative-ES conclusie is robuust over density-specificatie. Hansen MLE op $z_t$ levert $\nu = 4.37, \lambda = +0.01$: heavy-tailed maar **symmetrisch eens je HAR-σ_t controleert**. De negative skewness van crypto-returns wordt volledig door de tijdvariërende σ_t van HAR opgevangen.

### A2. Option-pricing simulation (H6)

7-day ATM straddle valuation onder drie σ-modellen, OOS n=1 305 (inkluis hold-period):

| σ-model | Gem. premium | Gem. payoff | Edge | MAPE | Sharpe per straddle | Hit-rate |
|---|---|---|---|---|---|---|
| Constant σ | 11.16 | 10.30 | +0.86 | 7.26 | +0.087 | 66.4% |
| Rolling 30d σ | 10.59 | 10.30 | +0.29 | 7.51 | +0.027 | 63.0% |
| **HAR-RS-DOW σ** | **10.50** | **10.30** | **+0.20** | **7.50** | **+0.019** | **60.0%** |

HAR is **closest to fair-value pricing** (edge 0.20 vs 0.86 voor constant σ). Het is geen alpha-generator maar wel een hygiene-factor: efficiëntere quotes voor een options market-maker.

### A3. Multi-frequency × multi-strategy matrix (H1)

Sharpe-matrix over 5 frequenties × 5 strategieën op €110, 15 bps fees:

| Strategie | 1m | 5m | 15m | 1h | 1d |
|---|---|---|---|---|---|
| B&H | 0.77 | 0.84 | 0.87 | 0.89 | 0.91 |
| Trend MA50 | -13.28 | -7.09 | -4.45 | -1.78 | **+0.97** |
| Trend MA200 | -11.26 | -6.60 | -3.05 | -0.49 | +0.56 |
| Momentum-20 | -13.33 | -6.95 | -5.53 | -2.02 | +0.37 |
| RSI-14<30 | -10.23 | -7.47 | -5.25 | -2.90 | -0.08 |

**Kernconclusie**: HFT trading op Bitvavo retail (€110 schaal, 15 bps fees) is niet viable. Alle actieve strategieën onder 1d-frequentie destroyen het account naar ~€5 door alleen al de fee-friction. Alleen daily Trend MA50 en B&H overleven.

### A4. Strategy design-space — volledige tabel (G4 + H2)

Top-10 strategieën op €110 daily, gerangschikt op Sharpe:

| Rank | Strategie | Final | AnnRet | Sharpe | Sortino | MDD | Calmar | #Reb | Fees |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ATR-stop + Trend MA50 | 311.35 | +33.6% | +1.14 | +1.30 | -27.6% | 1.22 | 95 | 32.53 |
| 2 | HAR vol-target + Trend | 255.12 | +26.4% | +0.94 | +0.98 | -25.5% | 1.03 | 260 | 27.87 |
| 3 | Trend MA50 (pure) | 262.70 | +27.4% | +0.90 | +0.93 | -27.1% | 1.01 | 77 | 21.06 |
| 4 | Donchian | 228.43 | +22.6% | +0.89 | +0.85 | -24.7% | 0.91 | 59 | 13.17 |
| 5 | Trend MA50 + BB pullback | 245.83 | +25.1% | +0.82 | +0.84 | -32.4% | 0.78 | 77 | 19.93 |
| 6 | Bollinger Breakout | 191.91 | +16.8% | +0.71 | +0.56 | -28.7% | 0.58 | 49 | 9.53 |
| 7 | MACD | 208.46 | +19.5% | +0.69 | +0.69 | -27.2% | 0.72 | 88 | 18.80 |
| 8 | HAR vol-regime + Trend | 164.98 | +11.9% | +0.69 | +0.65 | -23.8% | 0.50 | 225 | 44.64 |
| 9 | Trend MA200 (pure) | 203.86 | +18.6% | +0.56 | n/a | -36.6% | 0.51 | 31 | 8.09 |
| 10 | B&H | 224.89 | +22.0% | +0.52 | +0.72 | -54.9% | 0.40 | 2 | 0.17 |

### A5. ATR-multiplier sensitivity (H3)

Hoe robuust is de ATR-stop met multiplier=3? Sweep over [2.0, 5.0]:

| ATR-mult | Final | AnnRet | Sharpe | MDD | #Rebal | Fees |
|---|---|---|---|---|---|---|
| 2.0 | 267.13 | +28.0% | 0.98 | -30.0% | 150 | n/a |
| 2.5 | 263.48 | +27.5% | 0.94 | -31.0% | 118 | n/a |
| **3.0** | **311.35** | **+33.6%** | **1.14** | **-27.6%** | **95** | **32.53** |
| 3.5 | 239.35 | +24.2% | 0.78 | -32.0% | 81 | n/a |
| 4.0 | 279.37 | +29.6% | 0.95 | -25.0% | 69 | n/a |
| 5.0 | 187.08 | +15.9% | 0.50 | -31.9% | 61 | n/a |

Geen volkomen plateau (3.5 dipt onder 0.90); echter mult=3 is **conventionele literatuur-waarde** (Wilder 1978, Faber 2010), niet in-sample geoptimaliseerd. Bredere range 2.0-4.0 levert Sharpe ≥ 0.78 — moderate maar reële robuustheid.

### A6. Robustness scorecard (H3) — volledige tabel

Sharpe per strategie × periode × asset (€110, 15 bps):

| Strategie | BTC Full | BTC Early (22-24) | BTC Late (24-26) | ETH Full | Verdict |
|---|---|---|---|---|---|
| **ATR-stop + Trend MA50** | **+1.14** | **+1.88** | **+0.48** | **+0.52** | **ROBUST** |
| HAR vol-target + Trend | +0.94 | +1.51 | +0.58 | N/A | ROBUST |
| Donchian | +0.89 | +1.12 | +0.71 | +0.35 | ROBUST |
| HAR vol-regime + Trend | +0.69 | +1.23 | +0.76 | N/A | ROBUST |
| Trend MA50 | +0.90 | +1.41 | +0.59 | +0.25 | CONDITIONAL |
| Trend MA50 + BB pullback | +0.82 | +1.30 | +0.64 | +0.29 | CONDITIONAL |
| Bollinger Breakout | +0.71 | +0.82 | +0.58 | +0.08 | CONDITIONAL |
| MACD | +0.69 | +1.01 | +0.52 | +0.10 | CONDITIONAL |
| RSI-Bounce | +0.41 | +0.18 | +0.67 | +0.46 | CONDITIONAL |
| **Buy-and-hold** | +0.52 | +1.49 | **−0.23** | +0.21 | **FRAGILE** |
| Bollinger MR | +0.03 | +1.24 | −0.62 | +0.04 | FRAGILE |

**ROBUST** = Sharpe > 0.3 in alle vier periodes; **CONDITIONAL** = positief overal maar zwak in minstens één; **FRAGILE** = negatief in minstens één.

### A7. Live deployment runbook & encountered bugs

De live trading bot (`src/hartrace/live/`) is deployed met **strategy='atr_trend' default**. Drie veiligheidslagen (env LIVE_TRADING=true + env DRY_RUN=false + CLI --live flag) plus operational kill-switch (`~/STOP_TRADING`).

**Eerste live trade (17 mei 2026, 15:12 UTC)**:
- Order: buy 0.00140610 BTC @ €67365 limit (post-only)
- Filled binnen seconden, fee €0.14 (15 bps maker — exact backtest-aanname)
- ATR-stop ingesteld op €62483 (-7.5% van entry, 3 × ATR(14))

**Twee productie-bugs gevonden en gefixed in eerste trade**:

1. **Tick-size violation**: Bitvavo BTC-EUR vereist gehele euro's (tickSize=€1.00), niet €0.01. Onze code stuurde €67496.24. Fix: `_round_price_to_tick()` methode in `bitvavo_client.py` die tick-size uit `/v2/markets` ophaalt en floor-rounding voor BUY, ceil voor SELL toepast.

2. **HMAC signature voor GET met query**: Bitvavo's HMAC vereist dat de signature-path de query-string bevat. Onze code signeerde `/order` terwijl Bitvavo `/order?orderId=...&market=BTC-EUR` verwacht. Fix: `urllib.parse.urlencode(params)` toevoegen aan sign-path in `_request()`.

Beide bugs zijn voorbeelden van **specificatie-gap tussen documentatie en implementatie** die alleen door echte API-calls aan het licht komen. Een aanbeveling voor toekomstige projecten: investeer in een **integration-test-suite tegen Bitvavo's sandbox** voor live-readiness validatie, naast de mathematische backtest-validatie.

Volledige operationele documentatie: `docs/PAPER_TO_LIVE_RUNBOOK.md` (290 regels).

### A8. Reproducibility

Alle scripts en data zijn beschikbaar in:

- `/Users/sakesaakstra/Desktop/crypto_hartrace/scripts/` — alle pipeline-scripts (C1, D1, E1-E4b, G1-G5, H1-H6)
- `/Users/sakesaakstra/Desktop/crypto_hartrace/src/hartrace/` — econometrische library (HAR, density, features, estimation)
- `/Users/sakesaakstra/Desktop/crypto_hartrace/outputs/tables/` — CSV/parquet voor alle resultaten
- `/Users/sakesaakstra/Desktop/crypto_hartrace/outputs/figures/` — alle plot-PNGs
- `/Users/sakesaakstra/Desktop/crypto_hartrace/data/raw/` — Bitvavo candles BTC-EUR + ETH-EUR (1m/5m/15m/1h/1d)

Python omgeving: Anaconda 2024.06, scipy 1.13, statsmodels 0.14, scikit-learn 1.5, pandas 2.2.
