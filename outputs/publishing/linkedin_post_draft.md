# LinkedIn post — klaar om te publiceren

**Doel**: visibility voor quant-recruiters (Optiver, IMC, Flow Traders, Robeco, Aegon, model-validation teams)

---

## Versie 1 — Technisch (voor quant-audience)

🎓 Vandaag publiek gemaakt: mijn MSc Econometrie thesis-project — een complete pipeline voor BTC-EUR variantie-forecasting met HAR-RS-DOW.

Wat het doet:
↗ Out-of-sample CRPS = 0.498 over 1 312 dagen, statistisch dominant (DM p < 0.005) over 9 HAR-familie alternatieven
↗ Expected Shortfall-coverage Z1 = +1.73 (Acerbi-Szekely test) — Basel III-adequaat onder Normal, Student-t én Hansen skewed-t density-aannames
↗ Eerlijke decompositie: HAR als variance-forecaster levert waarde in risk-management en option-pricing, niet als directional trading signaal
↗ Live deployed op Bitvavo met 3-laags safety en eerlijk gedocumenteerde productie-bugs uit eerste live trade

Plus 20-passing pytest-suite voor reproducibility — wat in finance code helaas zeldzamer is dan zou moeten.

Code, thesis (PDF), defense deck en complete pedagogische walkthrough: https://github.com/SakeSaak/crypto-hartrace

#Econometrics #QuantitativeFinance #VolatilityForecasting #HARmodel #RiskManagement #BaselIII #VrijeUniversiteit

---

## Versie 2 — Verhalend (voor breder publiek + recruiters)

Eindelijk gepubliceerd waar ik het afgelopen jaar aan heb gewerkt — naast 32 uur per week bij Gasunie en mijn MSc Econometrie aan de VU.

De korte versie: *kun je BTC-volatiliteit beter voorspellen met econometrische modellen dan met simpele trend-volging?*

Het lange antwoord is nuancer:
✓ Voor risk-management: ja — HAR-modellen leveren Basel III-adequate Expected Shortfall-forecasts
✓ Voor option-pricing: ja — HAR-σ geeft fair-value prijzen dichter bij theoretisch optimum
✗ Voor directional trading: niet echt — ATR-stop met simpele MA50 verslaat alle HAR-variants in mijn backtest

De grootste les: een statistisch dominant model is niet automatisch economisch waardevol in elke toepassing. Het juiste econometrische model alignen met de juiste loss-function is wat publiceerbare research scheidt van werk-dat-er-goed-uitziet-maar-niet-houdt-buiten-sample.

Bonus inzicht: ik heb het systeem ook echt live gedeployed op Bitvavo en de eerste trade onthulde twee bugs die geen backtest ooit zou vinden (tick-size rounding en HMAC-signature met query strings). De gap tussen "backtest werkt" en "live trade slaagt" is groter dan papers suggereren.

Code en thesis-PDF: https://github.com/SakeSaak/crypto-hartrace

#Econometrics #DataScience #Trading #VrijeUniversiteit #MSc

---

## Versie 3 — Kort (voor X/Twitter)

Net publiek: mijn MSc thesis HAR-RS-DOW voor BTC-EUR variance forecasting.

Findings:
• CRPS 0.498 OOS (1312 days), DM-dominance p<0.005
• Expected Shortfall Z1 = +1.73 (Basel III adequate)
• Live deployed op Bitvavo, 2 production bugs gefixt

Code + 20 passing tests: https://github.com/SakeSaak/crypto-hartrace

---

## Posting strategy

1. **Versie 2** (verhalend) op LinkedIn — donderdag of vrijdag ochtend tussen 09:00-11:00 voor max EU-engagement
2. **Versie 1** (technisch) als comment op je eigen post, voor de details
3. **Versie 3** op X/Twitter als je dat gebruikt — tag @VUamsterdam en gebruik #econtwitter
4. Tag (na publicatie): je thesis-begeleider, eventueel @anthropic als je wil

## Verwachte response

Op basis van wat ik elders heb gezien voor vergelijkbare student-posts:
- 200-2000 LinkedIn impressions (afhankelijk van je network)
- Realistisch: 1-3 inbound recruiter-DMs binnen 2 weken
- Mogelijk: 1-2 "hoe heb je X gedaan" vragen van mede-studenten of jonge quants
