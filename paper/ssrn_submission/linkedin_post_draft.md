# LinkedIn Post Draft

## Versie 1 (formal, professional)

🎓 Pleased to share my MSc thesis working paper:

**"HAR-RS-DOW Variance Forecasting for Bitcoin: Statistical Dominance and Applications in Risk Management and Option Pricing"**

Key findings from 7 years of BTC-EUR data (with cross-asset validation on ETH-EUR, SOL-EUR):

📊 HAR-RS-DOW is the **singleton in the Model Confidence Set** (Hansen-Lunde-Nason 2011) at both 90% and 75% confidence levels — strongest possible statistical evidence of unique forecasting superiority within the HAR family.

📉 The model produces **Basel III-adequate Expected Shortfall** forecasts (Acerbi-Szekely Z1 ∈ [+1.73, +2.11]) across three density specifications.

💱 As input to Black-Scholes for 7-day ATM straddles, HAR-σ̂ yields the **smallest systematic edge** — operationally relevant for derivative market-makers.

🔬 Outperforms GARCH(1,1) benchmark by OOS R² margins of +0.54 to +0.79 on all three crypto assets.

Code, data, and full empirical pipeline available on GitHub:
https://github.com/SakeSaak/crypto-hartrace

SSRN preprint: [add SSRN link once available]

#Econometrics #QuantitativeFinance #Bitcoin #RiskManagement #BaselIII #RealizedVolatility #HARmodel

## Versie 2 (concise, hook-driven)

What's the empirically best variance forecaster for Bitcoin?

After estimating 10 HAR-family specifications on 2 334 days of BTC-EUR realized variance data and validating across ETH-EUR and SOL-EUR, the answer is HAR-RS-DOW — the Heterogeneous Autoregressive model with realized semivariance decomposition and day-of-week effects.

The Saturday volatility coefficient is significantly negative on all three cryptos (γ_Sat ∈ [-0.27, -0.19]), confirming weekend market microstructure as a structural feature.

The Hansen-Lunde-Nason (2011) Model Confidence Set contains only HAR-RS-DOW at 90% and 75% confidence on both CRPS and QLIKE.

Working paper + full code: https://github.com/SakeSaak/crypto-hartrace

#QuantitativeFinance #Bitcoin #Econometrics

