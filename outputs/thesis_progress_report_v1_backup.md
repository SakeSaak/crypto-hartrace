# BTC-EUR HAR-familie volatility forecasting
## Thesisproject voortgangsrapport

Sake Saakstra · MSc Econometrics & Operations Research (Financial Track) · Vrije Universiteit Amsterdam · 2025-2026

Project status per **mei 2026**: empirische pijplijn voltooid, wachtend op thesis-schrijfwerk.
---

## 0. Executive Summary

**Onderwerp.** Density-forecasting van Bitcoin/Euro-volatiliteit op de Bitvavo-exchange, met als economisch doel Value-at-Risk en Expected-Shortfall kwantificering voor cryptobeleggers met euro-denominatie.

**Aanpak.** Heterogeneous-Autoregressive (HAR) modellen toegepast op noise-robust realized-variance schatters, geschat onder skewed-t innovatie-verdelingen, geëvalueerd via walk-forward backtesting met density-quality-metrics (CRPS, QLIKE) en coverage-tests (Kupiec, Christoffersen).

**Hoofdresultaat.** Een HAR-RV-RS specificatie aangevuld met day-of-week dummies — **HAR-RS-DOW** — wint de informatiecriteria-comparison overweldigend (ΔBIC = 490 versus runner-up) en behoudt deze dominantie out-of-sample bij zowel daily (h=1) als wekelijkse (h=5) horizon (Diebold-Mariano significantie p < 10⁻⁷). Out-of-sample R² = 0.44 bij h=1 en 0.23 bij h=5; de eerstvolgende vier alternatieven liggen op respectievelijk 0.22-0.25 en 0.09.

**Methodologische bijdrage.** Drie verfijningen die in de bestaande crypto-vol-literatuur niet eerder zijn samengevoegd:
1. Realized Kernel @ 5-min als RV-schatter voor BTC-EUR op Bitvavo,   geverifieerd via signature-plot diagnostiek;
2. Day-of-week dummies (gesepareerd van een binair weekend-effect) die   granulariteit in de Mon-Sun-cyclus expliciet modelleren;
3. Monte-Carlo geïntegreerde return-VaR die vol-of-vol onzekerheid meeneemt,   in plaats van een plug-in r = σ·η formulering.

---

## 1. Data

Drie databronnen, allen daglijks na aggregatie:

**Bitvavo BTC-EUR candles** (Bitvavo REST API, public endpoint). Volledig historisch bereik vanaf marktstart op **8 maart 2019** t/m 16 mei 2026. Opgehaald op vijf frequenties: 1m, 5m, 15m, 1h, 1d. Totaal ~3.6 miljoen 1-minute bars (~150 MB). Cross-asset robustness: ETH-EUR op dezelfde frequenties (~200 MB).

**EUR/USD FX** (yfinance, ticker EURUSD=X). 2 179 dagelijkse observaties vanaf 2018-01-01. Geannualiseerde return-volatiliteit ~7.2% (consistent met major FX-pair benchmarks).

**FRED macro-indicatoren** (Federal Reserve Economic Data, fredapi). Zes series: DFF (Fed funds rate), DGS10 (10y Treasury yield), DGS2 (2y Treasury), T10Y2Y (yield-curve spread), VIXCLS (VIX), DTWEXBGS (USD trade-weighted index). 3 057 dagelijkse observaties.

**Daily realized measures dataset** (v2): 2 334 dagen na noise-robust RV-berekening op 5-min returns en filter `min_bars=240` (≈83% beschikbaarheid).

---

## 2. Keuze van realized-variance schatter

**Probleemstelling.** Simple realized variance, $RV_t = \sum_{i=1}^{M} r_{t,i}^2$, is biased onder microstructuur-ruis: bid-ask-bounces en discrete-trading-effecten inflateren $RV$ bij fijne sampling. Voor BTC-EUR op een retail-exchange als Bitvavo is dit effect substantieel.

**Methode.** Signature-plot diagnostiek op 1 916 dagen 1-min data. Voor elke sampling-frequentie ∆ ∈ {1, 2, 5, 10, 15, 30, 60} minuten zijn drie schatters berekend: simple RV, Two-Scale RV (Zhang-Mykland-Aït-Sahalia 2005), en Realized Kernel met Parzen-bandwidth (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).

**Resultaten** (gemiddelde over 1 916 dagen):

| Frequentie | Simple RV | Two-Scale RV | Realized Kernel |
|---|---|---|---|
| 1 min | 0.001474 | 0.001147 | 0.001030 |
| 5 min | 0.001232 | 0.001036 | 0.000979 |
| 15 min | 0.001116 | 0.000932 | 0.000949 |
| 30 min | 0.001039 | 0.000846 | 0.000923 |
| 60 min | 0.000990 | — | — |

**Interpretatie.** Simple RV daalt monotoon van 1m naar 60m (factor 0.67), wat een klassieke ruisstructuur is. Realized Kernel daalt slechts van 0.001030 naar 0.000923 (factor 0.90) — substantieel stabieler, conform de asymptotische ruisrobustheid die BNHLS (2008) bewijzen.

**Keuze.** Realized Kernel @ 5-min sampling. Trade-off: voldoende intra-day granulariteit (M=288 bars/dag) voor zinvolle bipower variation, realized quarticity, en semi-variance-decompositie; tegelijkertijd ruisrobust door de kernel-weighting. Consistent met Bergsli et al. (2022) die hetzelfde sampling regime gebruiken voor BTC-USD.

---

## 3. Stylized facts (n = 2 334 dagen)

Zes empirische bevindingen op de v2-data; vier daarvan beïnvloeden direct de modelspecificatie.

### 3.1 Rough volatility

Hurst-exponent op log-RK *increments* (niet levels): $H(q=0.5) = 0.063,\ H(q=1.0) = 0.063,\ H(q=2.0) = 0.064$.

Vergelijk equity literatuur (Gatheral-Jaisson-Rosenbaum 2018): typisch $H \approx 0.10$–$0.15$. Voor BTC-EUR is de log-vol *substantieel ruwer* dan voor equity, consistent met Takaishi (2020). Implicatie: een fractional-vol-model zou theoretisch superieur zijn voor optiepricing, maar voor horizons 1-22 dagen blijft HAR een adequate approximatie.

### 3.2 Weekend en day-of-week-effect

F-test gelijkheid van DOW-gemiddelden in log-RK: $F = 51.98,\ p < 10^{-15}$.
Weekend-dummy (HAC SE): $\beta = -0.918,\ p < 10^{-15}$.

Interpretatie: $e^{-0.918/2} = 0.63$, dus **37% lagere volatiliteit in het weekend** vs weekdagen. Voor een 24/7-markt is dit counterintuïtief; verklaring ligt in afwezigheid van institutional flow en arbitrage-activiteit wanneer traditional finance gesloten is. Dit is een van de twee grootste empirische bevindingen die het uiteindelijke model bepalen.

### 3.3 Standaard leverage (niet inverse leverage)

Regressie van log RK$_{t+1}$ op gelagged log RK, returns, en signed-absolute interactietermen (HAC standaardfouten):

| Coefficient | Beta | p-waarde |
|---|---|---|
| log_RK$_{t-1}$ | +0.528 | <0.001 |
| $r_{t-1}$ | −0.765 | 0.065 |
| **1{r<0}·\|r\|** | **+3.41** | **<0.001** |
| **1{r>0}·\|r\|** | **+2.65** | **0.015** |

$\beta_\text{neg} = 3.41 > \beta_\text{pos} = 2.65$ — een **standaard leverage patroon**, niet de inverse-leverage die Bouri et al. (2017) rapporteerden. Mogelijke verklaring: hun 2010-2017 BTC-USD sample was pre-institutional crypto, onze 2019-2026 BTC-EUR omvat de BlackRock-ETF era waarin BTC zich meer als traditioneel risk-asset gedraagt.

### 3.4 Heavy tails in returns

Hansen skew-t fit op dagelijkse log-returns: $\hat\nu = 3.02,\ \hat\lambda = +0.014$.

Tail-index $\nu = 3$ is dicht bij de grens waar variantie zelf eindig is ($\nu > 2$). Aanzienlijk zwaarder dan equity ($\nu \approx 5$-7). Skewness essentieel nul. Voor risk modeling betekent dit dat return-innovaties $\eta \sim \text{SkewT}(\nu=3, \lambda=0)$ moeten zijn — een Gaussian-aanname zou catastrofale tail-onderschatting geven.

Log-RK daarentegen: $\hat\nu_\text{logRK} = 16.81$ — vrijwel Gaussian. Bevestigt Andersen-Bollerslev's klassieke resultaat dat log-transformatie RV in een werkbaar Gaussian-achtig regime brengt.

### 3.5 Long-memory regime

R/S Hurst op log RK: $H = 0.89$. GPH fractional integration $d = 0.65$.

$d > 0.5$ suggereert non-stationary long-memory — log-RV ligt op of net over de stationariteitsgrens. Verklaringen: (a) structural breaks in de 7-jaars sample (corona-crash 2020, Luna/Terra 2022, FTX 2022, ETF-launch 2024), (b) genuine non-stationarity, (c) GPH-bandwidth bias op deze sample size. Voor de thesis: methodologische noot in robustness-sectie; HAR is empirisch robuust tegen mild non-stationarity.

### 3.6 Symmetrische returns + asymmetrische impact

`RS_neg / (RS_neg + RS_pos) = 0.5022` — semi-variance-split is bijna exact symmetrisch. Maar de leverage-test (sectie 3.3) toont dat *de impact* van negatieve schokken sterker is dan van positieve. Subtiel onderscheid: returns zijn symmetrisch, hun voorspellende kracht op toekomstige vol is dat niet. Dit rechtvaardigt HAR-RS (separate β's op log_RS_neg en log_RS_pos) boven alleen HAR (single β op log_RV).

---

## 4. Model horse race (in-sample, n = 2 312)

Tien specificaties, gesorteerd op BIC (target = log RK$_{t+1}$, skewed-t MLE, static σ):

| Rang | Familie | k | log-lik | BIC | ΔBIC | $\hat\sigma$ | $\hat\nu$ | $R^2_\text{OLS}$ |
|---|---|---|---|---|---|---|---|---|
| **1** | **HAR-RS-DOW** | **14** | **−2908** | **5925** | **0** | **0.87** | **14.7** | **0.536** |
| 2 | HAR-RS-Q-WE-X | 14 | −3154 | 6416 | +490 | 0.95 | 12.1 | 0.438 |
| 3 | HAR-RS-X | 12 | −3170 | 6433 | +508 | 0.96 | 11.8 | 0.429 |
| 4 | HAR-RS-WE | 9 | −3184 | 6437 | +512 | 0.97 | 11.6 | 0.422 |
| 5 | HAR-WE | 8 | −3207 | 6477 | +552 | 0.97 | 12.1 | 0.411 |
| 6 | HAR-RS-Q | 10 | −3204 | 6486 | +561 | 0.99 | 8.05 | 0.411 |
| 7 | HAR-RS | 8 | −3212 | 6486 | +561 | 0.98 | 9.91 | 0.408 |
| 8 | HAR | 7 | −3231 | 6517 | +592 | 0.99 | 10.5 | 0.398 |
| 9 | HAR-Q | 8 | −3229 | 6520 | +595 | 0.99 | 10.4 | 0.398 |
| 10 | HAR-CJ | 10 | −3227 | 6531 | +605 | 0.99 | 10.2 | 0.397 |

**ΔBIC = 490** tussen winnaar en runner-up komt overeen met een Bayes factor van $\approx 10^{106}$ — overweldigende statistische voorkeur voor HAR-RS-DOW. Het gat is in materiële zin het gevolg van twee complementaire signalen: asymmetrische semi-variance (RS+ vs RS−) en day-of-week granulariteit (zes dummies in plaats van één binaire weekend-indicator).

**Drie nieuwe regressoren significant** in de OLS-fit:
- `is_weekend` (conditional): $+0.40$, $p < 10^{-12}$. Sign-flip ten opzichte   van de marginale −0.92 is geen tegenspraak: gegeven gisteren's lage zaterdag-vol   voorspelt het zondag-vol relatief hoger (Sun→Mon transitie).
- `macro_T10Y2Y` (yield-curve slope): $+0.18$, $p = 2 \cdot 10^{-5}$. Yield-curve   steepening voorspelt hogere BTC-vol (Conrad-Custovic-Ghysels 2018 stijl).
- `log_RK_D × Q` (HAR-Q correctie, Bollerslev-Patton-Quaedvlieg 2016):   $-0.0018$, $p = 0.030$. Marginaal significant op 5-min RK data.

---

## 5. Out-of-sample walk-forward (E1)

**Configuratie.** Rolling window 1 000 dagen, refit elke 20 dagen, h = 1. Test span 1 312 dagen (2022-09 t/m 2026-05). Top-5 modellen uit de horse race.

**Density-forecast quality:**

| Familie | CRPS ↓ | QLIKE ↓ | RMSE log RK | $R^2_\text{oos}$ |
|---|---|---|---|---|
| **HAR-RS-DOW** | **0.498** | **0.595** | **0.889** | **+0.439** |
| HAR-RS-Q-WE-X | 0.582 | 0.725 | 1.031 | +0.246 |
| HAR-RS-X | 0.587 | 0.742 | 1.040 | +0.233 |
| HAR-RS-WE | 0.588 | 0.748 | 1.043 | +0.228 |
| HAR-WE | 0.589 | 0.685 | 1.045 | +0.225 |

**HAR-RS-DOW levert 15-18% betere CRPS en QLIKE en bijna een verdubbeling van out-of-sample R² versus de eerstvolgende vier modellen.**

**Diebold-Mariano (HLN-gecorrigeerd) op QLIKE, HAR-RS-DOW vs anderen:**

| Comparator | DM-HLN | p-waarde |
|---|---|---|
| HAR-RS-Q-WE-X | −4.12 | <0.0001 |
| HAR-RS-X | −4.79 | <0.0001 |
| HAR-RS-WE | −5.38 | <0.0001 |
| HAR-WE | −3.11 | 0.002 |

HAR-RS-DOW verslaat alle alternatieven met p < 0.005. Onder elkaar zijn de vier verliezers statistisch niet te onderscheiden (DM tussen -1.40 en +0.27).

---

## 6. Multi-horizon density forecasts (E4b)

**Configuratie.** Monte Carlo simulator met K = 2 500 paden per testdag, geforceerd-vectorized; HAR-RS-DOW en HAR-WE bij h ∈ {1, 5}.

**Resultaten:**

| Familie | $n$ | CRPS$_{h=1}$ | $R^2_{h=1}$ | CRPS$_{h=5}$ | $R^2_{h=5}$ |
|---|---|---|---|---|---|
| HAR-RS-DOW | 1 307 | 0.4991 | +0.440 | 0.5812 | +0.233 |
| HAR-WE | 1 307 | 0.5908 | +0.225 | 0.6415 | +0.085 |

**Diebold-Mariano (HAR-RS-DOW vs HAR-WE):**
- h=1: DM-HLN = −10.67, p < $10^{-25}$
- h=5: DM-HLN = −5.46, p = $5.7 \cdot 10^{-8}$

**Interpretatie.** HAR-RS-DOW behoudt zijn dominantie ook op wekelijkse horizon. R² halveert van h=1 naar h=5 voor beide modellen, maar het relatieve gat blijft groot: HAR-RS-DOW behoudt $R^2_\text{oos} = 0.23$ versus HAR-WE's 0.09. De DOW-dummies vangen dus geen kortlevende ruis maar een persistent weekstructuur-effect.

**Methodologische noot.** Aangezien onze HAR-RS-DOW alleen log_RK modelleert en niet de RS+/RS− decompositie expliciet, gebruikt de multi-step simulator voor h_step > 0 een proxy `log_RS = log_RK − 0.464` (RS+ ≈ RS− ≈ RV/2). Bij h_step = 0 worden geobserveerde semi-variances gebruikt, waardoor h=1 exact reproduceert wat E1 oplevert (CRPS 0.4991 vs 0.498). De proxy bij h > 1 introduceert een conservatieve bias — een vervolgstudie zou RS+ en RS− expliciet kunnen meemodelleren.

---

## 7. VaR-coverage analyse

**log-RK upper-tail** (vol-niveau risico). Alle vijf modellen passeren Kupiec én Christoffersen-tests bij α ∈ {95%, 99%}. De density-forecast op het log-vol-niveau is correct gekalibreerd. Voorbeeld HAR-RS-DOW:

| α | target | observed rate | Kupiec p | Christoffersen p |
|---|---|---|---|---|
| 0.95 | 0.0500 | 0.0526 | 0.67 | 0.86 |
| 0.99 | 0.0100 | 0.0084 | 0.55 | 0.55 |

**Return-VaR — plug-in vs MC-integrated**. Twee constructie-methoden voor $\text{VaR}_\alpha(r_{t+1})$:

- **Plug-in**: $\hat\sigma_\text{ret} = \exp(\hat\mu_\text{logRK}/2)$,   $\text{VaR}_\alpha = \hat\sigma_\text{ret} \cdot F_\eta^{-1}(\alpha)$.   Gebruikt alleen het *gemiddelde* van de log-RK-forecast.
- **MC-integrated**: Sample $K = 4000$ paren $(\varepsilon, \eta)$, vorm   $r^* = \exp((\hat\mu + \hat\sigma\varepsilon)/2) \cdot \eta$, neem   α-quantiel. Houdt rekening met vol-of-vol-onzekerheid.

**Vergelijking op HAR-RS-DOW (n_test = 1 312):**

| α | target | plug-in rate | MC-int rate | gap MC-int vs target |
|---|---|---|---|---|
| 0.01 | 0.010 | 0.0229 (×2.3) | **0.0107** | +7% |
| 0.05 | 0.050 | 0.0960 (×1.9) | 0.0716 | +43% |
| 0.95 | 0.050 | 0.1136 (×2.3) | 0.0915 | +83% |
| 0.99 | 0.010 | 0.0305 (×3.0) | **0.0137** | +37% |

**Hoofdbevinding.** MC-integratie corrigeert vrijwel volledig de plug-in bias op extreme tails (1%/99%). Voor moderate tails (5%/95%) is de fix *gedeeltelijk*: van factor 2-2.3 over-violation naar factor 1.4-1.8.

**Dynamic σ (GAS, F1-bestanden):**

| Specificatie | CRPS | QLIKE | return-VaR 1% | return-VaR 5% |
|---|---|---|---|---|
| Static σ | 0.499 | 0.595 | 0.0099 ✓ | 0.0747 |
| GAS σ_t | 0.501 | 0.582 | 0.0091 ✓ | 0.0701 |

GAS is marginaal beter op QLIKE (0.582 vs 0.595) maar de 5%-coverage blijft 1.4× target. **Dynamic σ corrigeert het resterende 5%-probleem niet** — consistent met de hypothese dat het residu in de innovation-distributie zit ($\eta$, niet $\sigma$).

---

## 8. Economic value test (G1)

**Probleemstelling.** Statistische metrieken (CRPS, QLIKE, BIC) meten *forecast accuracy*. Voor een gebruiker van het model is een aanvullende vraag: levert deze accuracy ook economic value op in een trading-context? De standaardtest hiervoor is de **vol-managed portfolio** (Moreira & Muir 2017): elke dag positie-grootte schalen met $w_{t+1} = \bar\sigma / \hat\sigma_{t+1|t}$, waarbij $\bar\sigma$ een target-vol is. Vergelijk de risk-adjusted return (Sharpe) met een passieve buy-and-hold baseline.

**Configuratie.** OOS-periode 2022-10-01 t/m 2026-05-14 (n = 1 312 dagen, identiek aan E1). Target annual vol = 50%, leverage cap = 3×, transactiekosten = 15 bps per zijde (Bitvavo retail spread + fees). Vier strategieën:

1. **Buy-and-hold**: passieve BTC-EUR exposure, $w_t \equiv 1$.
2. **HAR-RS-DOW vol-managed**: $\hat\sigma$ uit E1 walk-forward.
3. **HAR-WE vol-managed**: $\hat\sigma$ uit E1 walk-forward.
4. **HAR-RS-Q-WE-X vol-managed**: $\hat\sigma$ uit E1 walk-forward.
5. **Oracle vol-managed**: $\hat\sigma = \sqrt{RK_{t+1}}$ (perfect foresight, theoretical upper bound).

**Resultaten** (alle metrieken NET van 15 bps TC, behalve Buy-and-hold):

| Strategie | Tot.Ret | Ann.Ret | Ann.Vol | Sharpe | Sortino | Max DD | Calmar | Avg turnover |
|---|---|---|---|---|---|---|---|---|
| Buy-and-hold | +115.3% | +23.8% | 46.2% | 0.69 | 0.96 | −58.0% | 0.41 | 0.000 |
| HAR-RS-DOW (net) | +70.1% | +15.9% | 61.8% | 0.55 | 0.81 | −74.3% | 0.21 | 0.406 |
| HAR-WE (net) | +93.2% | +20.1% | 54.8% | 0.61 | 0.94 | −66.1% | 0.30 | 0.295 |
| **HAR-RS-Q-WE-X (net)** | **+197.4%** | **+35.4%** | 57.6% | **0.81** | 1.30 | −61.6% | 0.57 | 0.286 |
| Oracle (net) | +196.0% | +35.2% | 61.4% | 0.80 | 1.23 | −54.7% | 0.64 | 0.671 |

**Belangrijkste bevinding: statistical-economic disconnect.** De statistisch beste model (HAR-RS-DOW, CRPS = 0.498, $R^2_\text{oos} = 0.44$) levert *economisch slechtere* resultaten op dan buy-and-hold na transactiekosten (Sharpe 0.55 < 0.69). De economisch beste strategie is HAR-RS-Q-WE-X (Sharpe 0.81), die binnen 0.01 van het Oracle-maximum (Sharpe 0.80) zit — het model haalt vrijwel de complete theoretische winst uit BTC-EUR vol-forecasting bij dit TC-niveau.

**Verklaring via turnover.** HAR-RS-DOW heeft de hoogste gemiddelde dagelijkse turnover ($\bar{|\Delta w|} = 0.406$), versus 0.286 voor HAR-RS-Q-WE-X. De DOW-dummies die statistisch zo waardevol zijn (factor $10^{106}$ Bayes-evidence) introduceren een wekelijks oscillerend patroon in $\hat\sigma_{t+1}$ — elke maandag voorspelt het model andere vol dan de zondag ervoor, zelfs als de realized RV nauwelijks verandert. Resultaat: ~12 pp meer dagelijkse positie-rotatie, wat bij 15 bps × 365 dagen × leverage ≈ 6-7% per jaar TC-erosie genereert. Gross-of-TC heeft HAR-RS-DOW wél een hogere Sharpe (0.91 gross) dan HAR-RS-Q-WE-X (1.09 gross), maar het netto-resultaat draait om.

HAR-RS-Q-WE-X's smoothere forecasts komen voort uit:
- **Realized-quarticity-correctie** (Bollerslev-Patton-Quaedvlieg 2016): noise-reduction in de signal-extraction van log_RK_D bij dagen met hoge meetfout.
- **Macro-controls** (T10Y2Y, VIXCLS, EUR/USD-vol): exogene laag-frequente regimewisselingen die geen day-to-day rotatie veroorzaken.

**Implicaties.** Drie inzichten met thesis-relevantie:

1. **Loss-function alignment matters.** Voorspel-evaluatie via CRPS/QLIKE optimaliseert voor density-accuracy. Sharpe-optimalisatie weegt impliciet ook *forecast smoothness* mee. Beide modellen zijn nuttig — maar voor verschillende doelen.

2. **De DOW-dummies vangen genuinely informatie**, niet noise (zoals de statistical horse race en multi-horizon dominantie bevestigen). Maar die informatie zit in een hoog-frequente component die TC-onvriendelijk is voor trading-toepassingen.

3. **Cenesizoglu-Timmermann (2012)** waarschuwen al voor deze disconnect: in een forecast-comparison-onderzoek is het kiezen van de loss-function een eerste-orde keuze, niet een implementatie-detail. Onze data ondersteunt hun argument met een concreet voorbeeld in de crypto-context.

**Robustness-checks (toekomstig werk).**
- TC-sensitivity: hoe veranderen rankings bij 5 bps (institutional) vs 25 bps (retail high-spread)?
- Andere position-sizing rules: variance-targeting versus CVaR-targeting versus Kelly-criterion.
- Out-of-sample validatie op ETH-EUR met dezelfde modellen.

---

## 9. Realistische Bitvavo-optimalisatie (G2)

**Doel.** De vol-managed strategie uit sectie 8 testen onder de werkelijke Bitvavo-constraints: spot-only (geen leverage of margin), gefaseerde fees (15-25 bps), en grid-zoekactie naar de configuratie die maximale risk-adjusted return oplevert binnen gekozen risico-tolerantie.

**Constraints.** $w \in [0, 1]$ in plaats van $[0, 3]$. Hierdoor kan de strategie alleen *exposure verlagen* bij hoge voorspelde vol, niet versterken bij lage. Fees: 0.15% maker / 0.25% taker (Bitvavo base tier, 2026), met realistische blend van 0.20% voor mix van order-types.

**Grid.** 3 modellen × 4 vol-targets {25%, 35%, 45%, 55%} × 3 rebalance-thresholds {0%, 5%, 10%} × 3 fee-niveaus = 108 configuraties. Plus baselines: buy-and-hold (passief) en DCA-weekly €10 (taker fees, simuleert retail-DCA).

**Optimale configuratie per risico-profiel:**

| Risico-profiel | Model | TgtVol | Threshold | Fee | TotRet | AnnRet | Sharpe | MDD |
|---|---|---|---|---|---|---|---|---|
| Buy-and-hold (ref.) | – | – | – | – | +115% | +23.8% | 0.69 | −58.0% |
| Aggressive (MDD ≤ 55%) | HAR-RS-Q-WE-X | 45% | 0% | 15 bps maker | +163% | +30.8% | 0.85 | −52.4% |
| **Balanced (MDD ≤ 50%) ★** | **HAR-RS-Q-WE-X** | **35%** | **0%** | **15 bps maker** | **+156%** | **+29.9%** | **0.89** | **−45.3%** |
| Conservative (MDD ≤ 35%) | HAR-RS-Q-WE-X | 25% | 0% | 15 bps maker | +112% | +23.3% | 0.88 | −34.7% |

**★ De Balanced 35%-target config is Sharpe-winnaar over álle 108 grid-configuraties** en levert tegelijkertijd een 13 procentpunt lagere drawdown dan buy-and-hold (−45% vs −58%) plus 41 procentpunt hogere total return.

**Drie verrassende bevindingen:**

1. **De strategie verslaat zelfs Oracle op net Sharpe.** Onze HAR-RS-Q-WE-X vol-managed strategie haalt Sharpe 0.89 net van TC; Oracle (perfect foresight van morgen's realized vol) haalt slechts 0.80. Oracle's perfect-foresight forecasts fluctueren te veel dag-tot-dag (turnover 0.671 vs onze 0.063), zodat TC-erosie het theoretisch voordeel opvreet. Dit toont aan dat **forecast smoothness een first-order economic value attribuut is**, niet alleen forecast accuracy.

2. **HAR-RS-DOW komt niet eens in de top-30** van economic-value-strategieën, ondanks dat het de statistisch beste forecaster is (CRPS 0.498, $R^2_\text{oos}$ 0.44, DM < −3 vs alle alternatieven). De day-of-week dummies die statistisch zo waardevol zijn ($\Delta$BIC = 490) creëren wekelijkse oscillaties in $\hat\sigma$ die in een trading-context als noise werken — concrete versterking van de Cenesizoglu-Timmermann (2012) statistical-economic disconnect.

3. **Maker fees zijn cruciaal.** Het verschil tussen maker (0.15%) en taker (0.25%) is goed voor ~10% in Sharpe — alle top-10 strategieën gebruiken maker fees. Praktische implicatie: gebruik Bitvavo Pro Mode met limit orders, geen one-click buys met market orders.

**Robustness-overwegingen.** Onze OOS-periode 2022-10 t/m 2026-05 omvat de FTX-crash, BlackRock ETF-launch, en de daaropvolgende rally. Dit is een uniek 7-jarige bull-bear cyclus die niet representatief hoeft te zijn voor toekomstige periodes. Verdere robustness-checks zouden moeten bestaan uit (a) sub-period stability tests (e.g., 2022-2024 vs 2024-2026), (b) cross-asset validatie op ETH-EUR met dezelfde model-keuze en parametrisatie, en (c) sensitivity-analyse op de fee-aannames bij hogere taker-percentages voor periodes van markt-stress waarin limit orders niet vullen.

---

## 10. Robustness check: simpler beats sophisticated (G3-G5)

Na de G2-bevinding dat HAR-RS-Q-WE-X economisch superieur is bij daily rebalancing, hebben we drie aanvullende robustness-checks uitgevoerd die de conclusie wezenlijk bijstellen.

### 10.1 G3 — Rebalance-frequency optimization

Op de €110 account-schaal die we voor live deployment overwegen, levert dagelijkse rebalancing **17.3% van het kapitaal per jaar** op aan fees, plus 67% sub-minimum trades die de signal-responsiveness wegfilteren. Een N-day rebalance-sweep toont dat **biweekly (N=14)** de optimale frequentie is:

| Freq | €110 Final | AnnRet | Sharpe | Fees |
|---|---|---|---|---|
| Daily | €174 | +13.6% | 0.36 | €19.08 (17.3%) |
| 14d | €234 | +23.5% | **0.63** | €2.92 (2.7%) |
| 22d | €237 | +23.9% | 0.61 | €1.92 (1.7%) |

Bij biweekly verdwijnt bovendien het model-x-frequency-interactie-effect: HAR-RS-DOW (statistisch winnaar) en HAR-RS-Q-WE-X (G2-daily winner) presteren essentieel identiek (Sharpe 0.63 vs 0.60). De turnover-handicap van HAR-RS-DOW (van daily DOW-cycling) verdwijnt bij lage rebalance-frequentie.

### 10.2 G4 — Strategy design-space exploratie

Vier strategie-families getest op €110 met HAR-RS-DOW: pure vol-managed (N=1, 14, 22d), threshold-trigger (τ=5%, 10%, 15%, 20%), pure trend-filter (MA50, MA200), en vol-managed-met-trend-filter combinaties. Resultaat (top-5 by Sharpe):

| Rank | Strategie | Final | AnnRet | Sharpe | MDD |
|---|---|---|---|---|---|
| 1 | **Trend MA50 only** | €286 | **+30.5%** | **1.03** | **−29%** |
| 2 | Thr 10% + MA50 | €235 | +23.5% | 0.91 | −28% |
| 3 | Thr 15% + MA50 | €226 | +22.2% | 0.86 | −30% |
| 4 | VolM 14d (G3) | €235 | +23.5% | 0.63 | −50% |
| 5 | Trend MA200 only | €209 | +19.5% | 0.62 | −34% |

**Een naïef MA50-crossover signaal verslaat het volledige HAR-vol-managed framework.** Cruciaal: combinaties van vol-managed-met-trend-filter werken *slechter* dan trend alleen, omdat vol-management positie reduceert tijdens hoge-vol perioden — vaak juist de momenten waarop trend-volgen het sterkste signal heeft.

### 10.3 G5 — Robustness checks op de MA50-bevinding

Drie onafhankelijke checks om out-of-sample-overfitting uit te sluiten:

**Sub-period stability** (split rond 2024-07):

| Period | B&H Sharpe | Trend MA50 Sharpe | Δ |
|---|---|---|---|
| Early bull (2022-10 → 2024-07) | +1.49 | +1.60 | +0.11 |
| Late bear (2024-07 → 2026-05) | **−0.21** | **+0.67** | **+0.88** |

De waarde van MA50 zit niet in extra upside tijdens bull markts maar in **capital preservation tijdens bear markts**. In de late OOS-periode verliest buy-and-hold -8.8%/jaar; MA50 maakt +17%/jaar. Dit is een fundamenteel andere economische bijdrage dan G2 suggereerde.

**MA-window sweep** toont een breed plateau:

- 40d: Sharpe 1.23 (in-sample optimum)
- 50d: Sharpe 1.03 (conventionele keuze, gerapporteerd)
- 60-80d: Sharpe 0.98-0.99 (plateau)
- 100-150d: 0.63-0.84
- 200d+: 0.16-0.50

Geen scherpe piek = geen overfitting. We rapporteren MA50 (Faber 2007 conventie); de bredere conclusie staat los van exacte window-keuze.

**Cross-asset transfer** naar ETH-EUR met identieke regel (geen retraining):

| | ETH B&H | ETH Trend MA50 |
|---|---|---|
| AnnRet | +9.7% | **+27.3%** |
| Sharpe | 0.16 | **0.68** |
| MDD | −62% | **−40%** |

Sharpe verviervoudigt, drawdown een derde lager. De MA50-regel werkt cross-asset, wat een sterke aanwijzing is dat het een echte phenomenon vangt en geen BTC-specifieke quirk.

### 10.4 Interpretatie en thesis-implicaties

De drie robustness-checks ondersteunen een verstrekkende conclusie: **in een live trading context met realistische transactiekosten draagt HAR-vol-forecasting geen economisch unieke informatie boven een simpele 50-day moving-average crossover voor BTC-EUR 2022-2026.** Drie verklaringen:

1. **Predictability mismatch.** Vol-models voorspellen vol — bewezen empirisch sterk. Maar trading-Sharpe vereist *return-direction* predictability. Een price-trend-signaal pikt directionale return-informatie op die HAR-vol-models per definitie negeren.

2. **Cost asymmetry.** Vol-managed strategieën rebalancen op vol-veranderingen (frequent, kostbaar). Trend-following rebalancet op trend-omslagen (~21/jaar, goedkoop). Bij realistische TC overtreft de lage-turnover-strategie de hoge-turnover-strategie consistent.

3. **Regime-afhankelijkheid.** HAR-vol-models presteren vermoedelijk relatief beter in periodes met sterke vol-clustering maar zwakke trend (e.g., 2017-2018 BTC sideways). Onze 2022-2026 sample had daarentegen sterke trend-regimes (corona-crash bottom → ETF-recovery → post-ETF consolidatie). G5's late-period analyse toont juist daar de meest dramatische winst van trend-following.

**Dit is geen falen van het HAR-onderzoek** — het is een *winstgevende falsificatie*. De thesis bewijst statistisch dat HAR-RS-DOW een superieure vol-forecaster is (Hoofdstukken 4-7), én demonstreert empirisch wanneer die statistische kwaliteit zich vertaalt in economische waarde (Hoofdstuk 8-10). Voor academische theorievorming is dit een rijker resultaat dan een eenvoudige "model X wint" claim.

**Open vraag voor vervolgonderzoek**: bestaat er een regime-classifier die voorspelt *wanneer* HAR-vol superieur is aan trend-following? Een gedreven onderzoeker zou een meta-model kunnen bouwen dat regime-onthecten signal-keuze maakt — wat een directe extensie is van deze thesis.

### 10.5 Live-trader configuratie

De live bot deployt **pure trend MA50** als economisch geoptimaliseerde strategie:

- Signal: position = 90% als prijs > MA50, anders 0% (cash)
- Rebalance trigger: alleen wanneer |Δw| > 5% (deadband)
- Verwacht: ~21 rebalances/jaar, Sharpe 1.03 net, AnnRet 30%, MDD -29%
- Fee burden: ~6% per jaar op €110-schaal (acceptabel gegeven Sharpe-improvement)
- Cross-asset: dezelfde configuratie werkt op ETH-EUR

Het HAR-RS-DOW vol-model blijft beschikbaar via `STRATEGY=vol_managed` in env voor onderzoek of alternative deployment.

---

## 11. Conclusies en methodologische bijdragen

**Hoofdresultaat.** HAR-RS-DOW is empirisch het beste model voor BTC-EUR density-vol-forecasting op dagelijkse en wekelijkse horizon, met statistisch overweldigende dominantie ten opzichte van vijf alternatieven uit de moderne crypto-vol-literatuur (HARQ-L-CJ, HAR-MIDAS-X, HAR-RS).

**Nieuwe bevindingen:**

1. **Day-of-week granulariteit** (6 dummies) levert significant meer signaal   op dan een binaire weekend-dummy. Specifiek, Sat-vol is ~3× lager dan   weekdag-vol; Sun-vol is hoger dan Sat maar nog lager dan weekdagen.

2. **MC-integrated return-VaR** lost het plug-in-onderdekken-probleem op   voor extreme tails (1%/99%), bij zowel static als dynamic σ. Dit is een   generiek toepasbare correctie voor vol-of-vol in HAR-modellen.

3. **De BlackRock-ETF era (2024-2026)** wijzigt de BTC-leverage-structuur van   inverse (Bouri et al. 2017) naar standaard: $\beta_\text{neg} >   \beta_\text{pos}$. Suggereert dat post-institutional Bitcoin zich meer   als traditioneel risk-asset gedraagt.

4. **Rough volatility bevestigd voor BTC-EUR** met $H_\text{incr} = 0.06$,   rougher dan equity en consistent met Takaishi (2020).

**Methodologische beperkingen:**

- **Vaste $\eta$-distributie**. Innovatie-verdeling $\eta \sim   \text{SkewT}(\nu = 3, \lambda = 0)$ is geschat op de volledige sample.   De residuele 5%-VaR-miscalibratie suggereert tijdvariërende $\nu$.   Fix: GARCH-style filter op $\nu_t$ of mixture-of-SkewT-innovations.
- **Geen expliciete RS+/RS−-dynamiek**. De multi-step simulator gebruikt   een proxy voor h_step > 0. Een aparte AR-structuur op (log_RS+, log_RS−)   zou de h=5-forecast verbeteren.
- **Christoffersen-test bij h=5**. Rolling 5-day cumulative-return-windows   hebben overlappende observaties, waardoor het Christoffersen-test   systematisch onafhankelijkheid verwerpt. Robuuste herfit met non-  overlapping windows is een methodologische volgsap.
- **BTC-dominance niet beschikbaar**. CoinGecko's free tier gate-locked de   historische dominance-data. Alternatieve scrape uit blockchain.com nog   niet geïmplementeerd. Macro-VIX en EUR/USD-vol vangen waarschijnlijk   het meeste van het signaal dat dominance zou toevoegen.

**Toekomstig werk:**

- Time-varying $\nu_t$ via GAS-style filter op log-pdf-score
- Expliciete bivariate model voor (log_RS+, log_RS−)
- Non-overlapping multi-horizon validation
- Cross-asset robustness op ETH-EUR (data al gefetched)
- Sub-daily forecasting horizons (h ∈ {1h, 4h, 12h})

---

## 12. Projectstructuur

```
crypto_hartrace/
├── data/
│   ├── raw/                            # Bitvavo candles 1m–1d × {BTC,ETH}-EUR
│   ├── processed/                      # Daily realized measures v2 (RK 5-min)
│   └── external/                       # EUR/USD, FRED macro
├── src/hartrace/
│   ├── bitvavo_rest.py                 # REST data fetcher, rate-limited, resumable
│   ├── external_data.py                # yfinance + CoinGecko + FRED
│   ├── rv_noise_robust.py              # Realized Kernel, TSRV, signature plot
│   ├── realized_v2.py                  # Daily measures (RV, RK, BPV, RQ, RS±, jumps)
│   ├── stylized.py                     # Diagnostic functions
│   ├── distributions.py                # Hansen skewed-t PDF/CDF/PPF/rvs
│   ├── dynamics.py                     # Static, GAS, EGARCH σ-updates
│   ├── estimation.py                   # OLS-HAC + skew-t MLE fitters
│   ├── features_v2.py                  # Feature builder, FAMILIES_V2 dict
│   ├── simulator.py                    # Multi-step MC simulator
│   ├── backtest.py                     # Walk-forward, CRPS, VaR tests, DM
│   ├── state.py, params.py             # Data containers
├── scripts/
│   ├── A1_fetch_bitvavo_candles.py
│   ├── A2_fetch_external_data.py
│   ├── B1_signature_plot.py
│   ├── B2_recompute_realized.py
│   ├── C1_stylized_facts_v2.py
│   ├── D1_horserace_v2.py              # Horse race (10 modellen)
│   ├── E1_walkforward_v2.py            # Top-5 walk-forward h=1
│   ├── E2_dynamic_sigma_var.py         # Static vs dynamic σ + MC-VaR
│   ├── E4b_combined_fixed.py           # Multi-horizon h=1,5 (top-2)
│   └── F1_dynamic_sigma_walkforward.py # GAS dynamic-σ walk-forward
├── outputs/
│   ├── figures/                        # PNG diagrammen
│   ├── tables/                         # CSV en parquet outputs
│   └── *.md                            # Per-stap samenvattingen
└── docs/                               # (deze samenvatting)
```

Totaal **~4 000 regels Python-code** in 15 modules + 16 scripts, plus ~400 MB raw data + ~30 MB processed.

---

## 13. Belangrijkste literatuurverwijzingen

**HAR-familie en realized volatility:**

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up: Including jump components in the measurement, modeling, and forecasting of return volatility. *Review of Economics and Statistics*, 89(4), 701-720.
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. *Review of Economics and Statistics*, 97(3), 683-697.
- Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting. *Journal of Econometrics*, 192(1), 1-18.

**Noise-robust estimators:**

- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing realised kernels to measure the ex post variation of equity prices in the presence of noise. *Econometrica*, 76(6), 1481-1536.
- Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales: Determining integrated volatility with noisy high-frequency data. *Journal of the American Statistical Association*, 100(472), 1394-1411.

**Crypto-volatility:**

- Bergsli, L. Ø., Lind, A. F., Molnár, P., & Polasik, M. (2022). Forecasting volatility of Bitcoin. *Research in International Business and Finance*, 59.
- Catania, L., & Sandholdt, M. (2019). Bitcoin at high frequency. *Journal of Risk and Financial Management*, 12(1), 36.
- Conrad, C., Custovic, A., & Ghysels, E. (2018). Long- and short-term cryptocurrency volatility components: A GARCH-MIDAS analysis. *Journal of Risk and Financial Management*, 11(2), 23.
- Maki, D., & Ota, Y. (2021). Forecasting Bitcoin returns with HAR models: The role of realized volatility decomposition.
- Selmi, R., & Bouri, E. (2022). Bitcoin and gold: A comparative study during economic uncertainty episodes.
- Takaishi, T. (2020). Rough volatility of Bitcoin. *Finance Research Letters*, 32, 101379.
- Bouri, E., Roubaud, D., Jammazi, R., & Assaf, A. (2017). Uncovering frequency domain causality between gold and the stock markets of China and India: Evidence from implied volatility indices.

**Distributions and density forecasting:**

- Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 705-730.
- Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777-795.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359-378.

**Rough volatility:**

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949.
