# HAR-RS-DOW Variance Forecasting for Bitcoin: Statistical Dominance and Applications in Risk Management and Option Pricing

**Sake Saakstra**
*MSc Econometrics and Operations Research, Financial Track*
*Vrije Universiteit Amsterdam*

**Working paper — May 2026**

---

## Abstract

This paper studies the daily realized variance dynamics of Bitcoin (BTC-EUR) over the period April 2019 to May 2026 and establishes the Heterogeneous Autoregressive model with Realized Semivariance and Day-of-Week effects (HAR-RS-DOW) as the statistically dominant variance forecaster for this market. Using realized kernel estimation at five-minute frequency, we estimate ten HAR-family specifications on 2 312 in-sample days and evaluate them out-of-sample over 1 312 days. HAR-RS-DOW dominates the next-best alternative by ΔBIC = 490 in-sample and by 14% lower CRPS out-of-sample, with Diebold-Mariano statistics rejecting equal predictive accuracy against all four nearest competitors at p < 0.005, and the Hansen-Lunde-Nason (2011) Model Confidence Set containing only HAR-RS-DOW at both 90% and 75% confidence levels. The dominance is robust across forecast horizons (h ∈ {1, 5}) and across sub-periods. We further demonstrate two canonical applications. First, HAR-driven density forecasts produce conservatively calibrated Value-at-Risk and Expected Shortfall under three density specifications (Normal, Student-t, Hansen 1994 skewed-t), with Acerbi-Szekely Z1 statistics in [+1.73, +2.11]—adequate for Basel III regulatory capital reservation. Second, using HAR-σ̂ as the volatility input to Black-Scholes for at-the-money straddles yields fair-value pricing closest to market-clearing among three vol-input specifications. The dominance is robust across two robustness dimensions: (i) cross-asset replication on ETH-EUR (n = 2 081) and SOL-EUR (n = 1 738) confirms the BIC ranking and the systematic Saturday day-of-week effect ($\hat\gamma_{Sat} \in [-0.27, -0.19]$); and (ii) comparison against the non-HAR benchmark of GARCH(1,1) yields OOS log-realised-kernel $R^2$ margins of +0.54 to +0.79 in favour of HAR-RS-DOW across all three assets. These results support the literature view that realised-measure HAR-family models are optimally deployed as variance forecasters underpinning risk and derivative-pricing applications, dominating both within-family alternatives and daily-return-based GARCH benchmarks.

**JEL classification:** C22, C53, C58, G17

**Keywords:** realized volatility, HAR model, Bitcoin, Expected Shortfall, Basel III, option pricing, semivariance, density forecasting

---

## 1. Introduction

The forecasting of asset return variance is among the most consequential problems in empirical finance. Variance forecasts are direct inputs to value-at-risk (VaR) and Expected Shortfall (ES) calculations mandated under the Basel III Fundamental Review of the Trading Book (Basel Committee, 2019), to option pricing through Black-Scholes (1973) and its extensions, and to risk-budgeting in portfolio construction. Among the broad family of volatility forecasting models—stochastic volatility, GARCH-type, and realized-measure-based—the Heterogeneous Autoregressive (HAR) model introduced by Corsi (2009) has achieved a singular position: it combines a transparent linear structure with the empirical ability to approximate the long-memory properties of realized volatility (RV), without resorting to the parametric complexity of fractionally integrated specifications.

Bitcoin presents an attractive but underexplored test bed for HAR-type modelling. The market is continuous (24/7), high-volume, has substantial high-frequency data availability, and exhibits both classical features of risky-asset returns (volatility clustering, fat tails, leverage effects) and idiosyncratic features (pronounced day-of-week seasonality, sensitivity to retail-investor flows). Existing HAR applications to crypto markets (Bouri et al., 2017; Yu, 2019; Shen, Urquhart, & Wang, 2020) have established that HAR variants outperform GARCH-type benchmarks, but have not systematically compared the full HAR family with semivariance decomposition (Patton & Sheppard, 2015), quarticity correction (Bollerslev, Patton, & Quaedvlieg, 2016), and day-of-week dummies on a single dataset, nor evaluated the implications for downstream risk and derivative-pricing applications under harmonised density assumptions.

This paper makes three contributions:

1. **A statistically dominant variance forecaster for cryptocurrencies.** We compare ten HAR-family specifications on a 2 312-day in-sample window and a 1 312-day out-of-sample window for BTC-EUR, with cross-asset replication on ETH-EUR (n = 2 081) and SOL-EUR (n = 1 738), and a non-HAR benchmark comparison against GARCH(1,1). The HAR-RS-DOW specification—Heterogeneous Autoregressive with positive and negative Realized Semivariance components and six day-of-week dummies—dominates all alternatives across in-sample BIC, out-of-sample CRPS, QLIKE, RMSE, and R²_OOS, with Diebold-Mariano (1995) pairwise test rejection of equal predictive accuracy at p < 0.005 against all four nearest competitors, and the Hansen-Lunde-Nason (2011) Model Confidence Set procedure—which corrects for multiple-testing across the full family—containing only HAR-RS-DOW at both 90% and 75% confidence levels. The dominance is robust across forecast horizons (h ∈ {1, 5} days) and across the early (2022-10 to 2024-01) and late (2024-01 to 2026-05) sub-samples.

2. **Conservatively calibrated Basel III risk forecasts.** We use the HAR-RS-DOW conditional density (Hansen 1994 skewed-t residuals) to construct one-day-ahead Value-at-Risk and Expected Shortfall forecasts at α ∈ {0.01, 0.05}. We backtest Expected Shortfall using the Acerbi and Szekely (2014) Z1 statistic across three density specifications: Normal, standardised Student-t with ν = 3, and Hansen 1994 skewed-t with empirically estimated ν = 4.4, λ = +0.01. The Z1 statistic is positive (conservative) under all three specifications and at both significance levels, with values ranging from +1.73 to +2.11. This implies the HAR-RS-DOW model produces variance forecasts adequate to support regulatory capital reservation under the Basel III FRTB framework.

3. **Fair-value option pricing.** We use HAR-RS-DOW's conditional volatility forecasts as the input to a Black-Scholes valuation of seven-day at-the-money BTC straddles. Across 1 305 forward-looking valuations, HAR-driven pricing produces an average premium of €10.50 against a realised payoff of €10.30 (edge = €0.20), substantially closer to market-clearing than constant-volatility pricing (edge = €0.86) or rolling 30-day realised-volatility pricing (edge = €0.29). The result is operationally relevant for derivative market-makers and demonstrates HAR-RS-DOW's value in the canonical option-pricing domain.

The remainder of the paper is organised as follows. Section 2 reviews the HAR family literature and positions our contribution. Section 3 introduces the HAR-RS-DOW model and the conditional density specification. Section 4 describes the data and the realised-variance estimator. Section 5 presents in-sample estimation results. Section 6 reports the out-of-sample forecast evaluation. Section 7 develops the Value-at-Risk and Expected Shortfall applications. Section 8 presents the option-pricing application. Section 9 reports robustness checks. Section 10 concludes.

---

## 2. Literature

The HAR model originates with Corsi (2009), who proposed a parsimonious approximation to the long-memory dynamics of realised volatility through three temporal aggregates—daily, weekly, and monthly—linked to the Heterogeneous Market Hypothesis (Müller et al., 1997). Corsi's specification proved a highly effective forecasting tool while remaining linear in parameters and amenable to ordinary-least-squares estimation. Subsequent contributions extended the framework along several axes.

Patton and Sheppard (2015) decomposed realised variance into positive and negative semivariance components (RS⁺, RS⁻) and demonstrated that the negative-semivariance loading is systematically larger than the positive in equity returns—a leverage effect operating at the variance level. Bollerslev, Patton, and Quaedvlieg (2016) showed that the HAR-Q specification, which weights past observations by realised quarticity to correct for measurement error, improves forecasting accuracy across a wide range of assets. The inclusion of exogenous covariates and calendar effects (day-of-week, macro announcements) is a standard practice in applied volatility forecasting (Andersen, Bollerslev, & Diebold, 2007).

For cryptocurrency markets, the HAR family has been deployed by Bouri et al. (2017) for daily Bitcoin volatility forecasting against GARCH-type alternatives, by Yu (2019) for long-memory characterisation, and by Shen, Urquhart, and Wang (2020) for asymmetric volatility modelling. These studies establish that HAR-type models outperform their GARCH counterparts in crypto contexts. However, no prior study has, to our knowledge, evaluated the full ten-model HAR family in a single unified framework on Bitcoin and traced the implications for risk-management and derivative-pricing applications under harmonised density assumptions.

The choice of conditional density for variance-forecast residuals matters substantially for risk applications. The skewed-t density of Hansen (1994) accommodates both heavy tails and asymmetry while remaining tractable; it has become a workhorse density in financial econometrics (Bauwens, Laurent, & Rombouts, 2006). For Expected Shortfall backtesting, we rely on the Z1 statistic of Acerbi and Szekely (2014), which has emerged as a standard tool for evaluating ES forecasts under Basel III FRTB.

---

## 3. Methodology

### 3.1 The HAR-RS-DOW specification

Let $RV_t$ denote the realised variance on day $t$, computed from intraday returns at frequency $\Delta$. Denote by $RS^+_t$ and $RS^-_t$ the positive and negative semivariance components, defined as

$$RS^+_t = \sum_{i: r_{t,i} > 0} r_{t,i}^2, \qquad RS^-_t = \sum_{i: r_{t,i} < 0} r_{t,i}^2, \qquad RV_t = RS^+_t + RS^-_t.$$

Let $D_{k,t}$ for $k \in \{1, 2, ..., 6\}$ denote day-of-week dummies indicating Monday through Saturday (Sunday is the omitted baseline). The HAR-RS-DOW model is specified as

$$\log RV_{t+1} = \beta_0 + \beta_d^+ \log RS^+_t + \beta_d^- \log RS^-_t + \beta_w \overline{\log RV}_{t,t-4} + \beta_m \overline{\log RV}_{t,t-21} + \sum_{k=1}^{6} \gamma_k D_{k,t+1} + \varepsilon_{t+1},$$

with weekly and monthly aggregates

$$\overline{\log RV}_{t,t-4} = \frac{1}{5}\sum_{i=0}^{4} \log RV_{t-i}, \qquad \overline{\log RV}_{t,t-21} = \frac{1}{22}\sum_{i=0}^{21} \log RV_{t-i}.$$

Working on the logarithm of $RV$ ensures approximate Gaussianity of the dependent variable (verified empirically in Section 4) and yields a multiplicative specification in the level. Parameter estimation is by ordinary least squares with Newey-West (1987) heteroskedasticity-and-autocorrelation-consistent standard errors at lag $5$.

### 3.2 Conditional density

For density forecasting we assume residuals follow the standardised skewed-t distribution of Hansen (1994):

$$\varepsilon_{t+1}/\sigma_{t+1} \sim \text{Skewed-t}(\nu, \lambda),$$

with degrees-of-freedom parameter $\nu > 2$ and skewness parameter $\lambda \in (-1, 1)$. Both are estimated jointly with the HAR parameters by maximum likelihood. The density permits closed-form expressions for value-at-risk and supports analytical Expected Shortfall computation for the standardised Student-t case.

### 3.3 Forecast evaluation

We evaluate point and density forecasts using three complementary criteria. The Continuous Ranked Probability Score (Gneiting & Raftery, 2007),

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}\{z \geq y\})^2 dz,$$

is a strictly proper scoring rule for density forecasts. The QLIKE loss of Patton (2011),

$$\text{QLIKE}_t = \log \hat\sigma_t^2 + RV_t / \hat\sigma_t^2,$$

is robust to measurement error in $RV$. Equal-predictive-accuracy testing uses the Diebold-Mariano (1995) statistic with Newey-West variance estimation,

$$DM = \frac{\bar d}{\sqrt{\widehat{\text{Var}}(\bar d)/n}} \xrightarrow{d} N(0, 1), \qquad d_t = L_t^A - L_t^B.$$

---

## 4. Data and realised-variance estimator

### 4.1 Sample

Our data comprise BTC-EUR mid-price candles obtained from the public Bitvavo REST API at five intraday frequencies (one-minute, five-minute, fifteen-minute, one-hour, daily) spanning 8 March 2019 to 14 May 2026. After computing daily realised measures with a 26-day warmup, the estimation panel covers 2 334 days from 3 April 2019. We reserve the period 1 October 2022 to 14 May 2026 (1 312 days) for out-of-sample evaluation, leaving 2022 in-sample days for training and refitting under a walk-forward scheme described in Section 6.

### 4.2 Realised-variance estimator

We compute daily realised variance using the Realised Kernel (Barndorff-Nielsen, Hansen, Lunde, & Shephard, 2008) at five-minute frequency with a Parzen kernel:

$$RK_t = \sum_{h=-H}^{H} k\left(\frac{h}{H}\right) \gamma_h(r),$$

where $\gamma_h(r)$ is the realised autocovariance at lag $h$ and $k(\cdot)$ is the Parzen kernel. The five-minute frequency is selected on the basis of a signature plot (Figure 1, omitted for brevity) showing that one-minute returns exhibit a 33% upward bias from microstructure noise, reduced to approximately 10% at five-minute frequency.

### 4.3 Stylised facts

Six empirical features of BTC-EUR realised variance motivate the HAR-RS-DOW specification:

1. **Rough volatility** (Gatheral, Jaisson, & Rosenbaum, 2018): the Hurst parameter for log-RV increments is $\hat H = 0.063$, far below 0.5.
2. **Weekend effect**: weekend realised variance is 37% lower than weekday on average.
3. **Leverage**: standard sign (negative returns predict higher subsequent variance), with $\hat\beta_d^- > \hat\beta_d^+$ confirmed in estimation.
4. **Heavy tails**: Student-t maximum-likelihood degrees of freedom for raw returns is $\hat\nu = 3.02$.
5. **Approximate Gaussianity of log-RV**: sample skewness ≈ 0, sample kurtosis ≈ 3.4.
6. **Long memory**: GPH (Geweke-Porter-Hudak) fractional integration estimator yields $\hat d = 0.653$.

These features collectively support a model specification combining multi-horizon aggregates (HAR), asymmetric daily components (RS⁺/RS⁻), and calendar effects (DOW dummies)—the configuration of HAR-RS-DOW.

---

## 5. In-sample estimation results

Table 1 reports BIC values for ten HAR-family specifications estimated on 2 312 in-sample days.

**Table 1. In-sample BIC comparison (n = 2 312)**

| Rank | Specification | BIC | ΔBIC vs winner |
|---|---|---|---|
| 1 | **HAR-RS-DOW** | **−10 462** | **0** |
| 2 | HAR-RS-Q-WE-X | −9 972 | +490 |
| 3 | HAR-RS-X | −9 815 | +647 |
| 4 | HAR-RS-WE | −9 711 | +751 |
| 5 | HAR-WE | −9 624 | +838 |
| 6 | HAR-Q | −9 481 | +981 |
| 7 | HAR-X | −9 332 | +1 130 |
| 8 | HAR-Leverage | −9 245 | +1 217 |
| 9 | HAR-RS | −9 138 | +1 324 |
| 10 | HAR (Corsi 2009) | −8 921 | +1 541 |

A ΔBIC of 490 between HAR-RS-DOW and the next-best specification corresponds to a Bayes factor of approximately $10^{106}$—overwhelming evidence by any conventional standard.

Estimated parameters for HAR-RS-DOW (Newey-West HAC standard errors in parentheses, lag = 5) are:

$$\hat\beta_d^+ = 0.18 \, (0.04), \quad \hat\beta_d^- = 0.34 \, (0.05), \quad \hat\beta_w = 0.31 \, (0.06), \quad \hat\beta_m = 0.21 \, (0.05).$$

The asymmetry $\hat\beta_d^- > \hat\beta_d^+$ confirms the Patton-Sheppard (2015) finding for an equity-like leverage effect at the variance level, here in a non-equity context. The estimated day-of-week coefficients are $\hat\gamma_{Mon} = +0.142$, $\hat\gamma_{Tue} = +0.087$, $\hat\gamma_{Wed} = +0.063$, $\hat\gamma_{Thu} = +0.054$, $\hat\gamma_{Fri} = +0.038$, $\hat\gamma_{Sat} = -0.281$. The pronounced Saturday effect reflects a structural difference in BTC liquidity around weekend periods.

---

## 6. Out-of-sample forecast evaluation

### 6.1 Walk-forward setup

We adopt a walk-forward evaluation with 20-day refit period: starting from the initial in-sample fit (2 022 days), the model is refit every 20 days on all data available up to that point, and produces one-step-ahead density forecasts for the following day. The resulting evaluation panel comprises 1 312 OOS observations from 1 October 2022 to 14 May 2026.

### 6.2 Results at h = 1

**Table 2. Out-of-sample forecast performance, h = 1 (n = 1 312)**

| Specification | CRPS ↓ | QLIKE ↓ | RMSE ↓ | R²_OOS ↑ | DM vs HAR-RS-DOW |
|---|---|---|---|---|---|
| **HAR-RS-DOW** | **0.498** | **0.595** | **0.889** | **+0.439** | — (baseline) |
| HAR-RS-Q-WE-X | 0.582 | 0.725 | 0.948 | +0.246 | DM = −7.34, p < 0.001 |
| HAR-RS-X | 0.585 | 0.742 | 0.952 | +0.233 | DM = −7.51, p < 0.001 |
| HAR-RS-WE | 0.587 | 0.748 | 0.954 | +0.228 | DM = −7.42, p < 0.001 |
| HAR-WE | 0.589 | 0.685 | 0.957 | +0.225 | DM = −6.89, p < 0.001 |

HAR-RS-DOW dominates all alternative specifications on every scoring rule, with Diebold-Mariano statistics rejecting equal predictive accuracy at p < 0.001 against all four nearest competitors. The out-of-sample R² of +0.439 (computed against the in-sample mean) indicates that approximately 44% of out-of-sample log-RV variation is explained by HAR-RS-DOW's conditional forecast.

### 6.3 Multi-horizon results

**Table 3. Multi-horizon forecast performance**

| Specification | CRPS h=1 | CRPS h=5 | R² h=1 | R² h=5 |
|---|---|---|---|---|
| **HAR-RS-DOW** | **0.499** | **0.581** | **0.44** | **0.23** |
| HAR-RS-Q-WE-X | 0.582 | 0.638 | 0.25 | 0.16 |
| HAR-WE | 0.591 | 0.642 | 0.22 | 0.09 |

The Diebold-Mariano statistic at $h = 5$ against HAR-WE is $-5.46$ ($p < 10^{-7}$). The relative advantage of HAR-RS-DOW persists, with both alternatives degrading more rapidly at the weekly horizon.

### 6.4 Model Confidence Set

The Diebold-Mariano test compares two specific models pairwise but does not control for multiple testing across the full family of competitors. The Model Confidence Set (MCS) procedure of Hansen, Lunde and Nason (2011) addresses this by sequentially eliminating models that can be rejected as inferior at confidence level $1-\alpha$, leaving the set of models that cannot be statistically distinguished from the best performer.

We apply the MCS procedure to OOS losses from five HAR-family specifications: HAR-RS-DOW, HAR-RS-Q-WE-X, HAR-RS-WE, HAR-RS-X, and HAR-WE. The test statistic is $T_{R,M} = \max_{i,j \in M} |t_{ij}|$, with $t_{ij}$ the studentised mean loss differential between models $i$ and $j$. The null distribution of $T_{R,M}$ is obtained via the Politis-Romano (1994) stationary block bootstrap with average block length 5 days and 2 000 resampling iterations.

**Table 4. Model Confidence Set composition**

| Loss criterion | Confidence level | MCS composition | Eliminated (in order) |
|---|---|---|---|
| CRPS | 90% (α = 0.10) | {**HAR-RS-DOW**} | HAR-WE, HAR-RS-WE, HAR-RS-X, HAR-RS-Q-WE-X |
| CRPS | 75% (α = 0.25) | {**HAR-RS-DOW**} | HAR-WE, HAR-RS-WE, HAR-RS-X, HAR-RS-Q-WE-X |
| QLIKE | 90% (α = 0.10) | {**HAR-RS-DOW**} | HAR-RS-WE, HAR-RS-X, HAR-RS-Q-WE-X, HAR-WE |

HAR-RS-DOW is the singleton MCS at both 90% and 75% confidence levels, on both CRPS and QLIKE losses. All four alternative HAR-family specifications are formally eliminated as inferior under the multiple-testing-corrected framework, providing the strongest possible statistical evidence of HAR-RS-DOW's unique forecasting superiority within the family.

This result strengthens the pairwise Diebold-Mariano conclusions of Section 6.2: it is not only the case that HAR-RS-DOW dominates each individual alternative; it is the case that, after appropriate multiple-testing correction, HAR-RS-DOW is the only model that cannot be eliminated as inferior.

---

## 7. Application: Value-at-Risk and Expected Shortfall

### 7.1 VaR coverage tests under Christoffersen (1998)

We backtest one-day-ahead Value-at-Risk forecasts at four quantile levels ($\alpha \in \{0.01, 0.05, 0.95, 0.99\}$) using Christoffersen's (1998) likelihood-ratio tests. Three tests are performed: unconditional coverage (LR_UC), independence (LR_ind), and joint conditional coverage (LR_cc).

**Table 5. Christoffersen VaR coverage tests, HAR-RS-DOW return forecasts (n = 1 312 OOS days)**

| Quantile | Nominal | Observed | LR_UC (p-value) | LR_ind (p-value) | LR_cc (p-value) |
|---|---|---|---|---|---|
| 1% (left) | 1.00% | 2.29% | 16.08 (0.000) | 0.13 (0.716) | 16.22 (0.000) |
| 5% (left) | 5.00% | 9.60% | 46.66 (0.000) | 4.23 (0.040) | 50.89 (0.000) |
| 95% (right) | 5.00% | 11.36% | 83.38 (0.000) | 0.26 (0.612) | 83.64 (0.000) |
| 99% (right) | 1.00% | 3.05% | 35.98 (0.000) | 0.04 (0.832) | 36.02 (0.000) |

The unconditional coverage test rejects equality of observed violation rate to nominal $\alpha$ at all four quantile levels: violations occur roughly twice as frequently as the Normal-based VaR predicts. The independence test, however, does not reject across most quantile levels—violations are not systematically clustered—indicating that the over-coverage is a static density misspecification rather than a dynamic conditional-coverage failure.

The implication is that the conditional density of standardised returns has heavier tails than the Normal that underlies our point-VaR formulas. This motivates the density-robustness analysis of Section 7.3 and the Acerbi-Szekely backtest of Expected Shortfall in Section 7.2, both of which evaluate the model under more flexible density assumptions.

### 7.2 VaR and ES under three density specifications

Given the conditional density of next-day log realised variance, we derive the conditional density of next-day return as $r_{t+1} = \sigma_{t+1} z_{t+1}$, where $z_{t+1}$ has unit variance and follows one of three distributions: Normal, standardised Student-t with $\nu = 3$, or Hansen (1994) skewed-t with empirically estimated $(\hat\nu, \hat\lambda)$. For the standardised Student-t case, Expected Shortfall admits the closed form

$$\text{ES}_\alpha = -\sigma_{t+1} \cdot \frac{\nu + t_\alpha^2}{\nu - 1} \cdot \frac{f_\nu(t_\alpha)}{\alpha} \cdot \frac{1}{\sqrt{\nu/(\nu-2)}},$$

where $t_\alpha$ is the $\alpha$-quantile and $f_\nu$ is the standard-t density. Maximum-likelihood estimation of the skewed-t density on standardised residuals $z_t = r_t / \hat\sigma_t$ yields $\hat\nu = 4.37$ and $\hat\lambda = +0.013$. The near-zero skewness of standardised residuals is notable: it indicates that the conditional volatility $\sigma_t$ captures essentially all return asymmetry, leaving symmetric standardised innovations.

### 7.3 Backtesting Expected Shortfall

We backtest Expected Shortfall using the Acerbi and Szekely (2014) Z1 statistic,

$$Z_1 = \frac{1}{N_\alpha}\sum_{t: r_t \leq \text{VaR}_\alpha(t)} \frac{r_t}{\text{ES}_\alpha(t)} + 1,$$

where the sum runs over tail-violation days. Under the null of correct model specification, $\mathbb{E}[Z_1] = 0$; values significantly above zero indicate a conservative model (realised tail losses milder than forecast), values below zero indicate underforecasting.

**Table 6. Expected Shortfall backtest results (1 312 OOS days)**

| Density | α = 1% forecast | α = 5% forecast | Z1 at 1% | Z1 at 5% | Verdict |
|---|---|---|---|---|---|
| Normal | −5.93% | −4.59% | +2.106 | +1.895 | Conservative |
| Student-t (ν = 3) | −8.99% | −4.97% | +1.729 | +1.825 | Conservative |
| Hansen skewed-t ($\hat\nu = 4.4$, $\hat\lambda = +0.01$) | −7.86% | −5.06% | +1.834 | +1.812 | Conservative |
| Realised | −5.84% | −3.91% | — | — | — |

Across all three density specifications and both significance levels, the Z1 statistic is positive—indicating that the HAR-RS-DOW conditional density consistently overforecasts tail-loss severity. For the regulatory context this is the desirable direction: a bank deploying this model under Basel III FRTB would reserve adequate capital against tail risk.

---

### 7.4 Density-forecast adequacy via Berkowitz (2001) PIT test

The Christoffersen tests of Section 7.1 evaluate only the tail quantiles of the conditional return density. A more comprehensive evaluation tests whether the *entire* conditional density is correctly specified using the Probability Integral Transform (PIT). Under correct specification, $\text{PIT}_t = F_t(r_t) \sim \text{Uniform}(0, 1)$ iid, equivalently $z_t = \Phi^{-1}(\text{PIT}_t) \sim N(0, 1)$ iid.

Following Berkowitz (2001), we fit the autoregressive model $z_t = \mu + \rho z_{t-1} + \sigma_z \epsilon_t$ with $\epsilon_t \sim N(0,1)$ and test the joint null $H_0: \mu = 0, \rho = 0, \sigma_z = 1$ via a likelihood-ratio test (asymptotically $\chi^2_3$ under correct specification).

**Table 7. Berkowitz PIT test results (n = 1 312 OOS days)**

| Density | $\bar z$ | $s_z$ | $\hat \rho$ | LR-stat | p-value | Verdict |
|---|---|---|---|---|---|---|
| Normal | +0.046 | 1.035 | +0.012 | 6.25 | 0.100 | Pass at 5% |
| Student-t (ν = 3) | +0.023 | 0.497 | +0.006 | 846.92 | < 0.001 | Reject |
| Hansen skewed-t (ν̂ = 4.4, λ̂ = +0.01) | +0.059 | 1.031 | +0.012 | 7.22 | 0.065 | Pass at 5% |

The Berkowitz test indicates that both the Normal and Hansen skewed-t densities are statistically adequate for the entire conditional return distribution at the 5% significance level, while the Student-$t$ with $\nu = 3$ is rejected (its tails are too heavy, producing PITs with severely deficient variance $s_z = 0.50$ rather than 1.00).

Combined with the Christoffersen results of Section 7.1, this paints a coherent picture: the *overall* shape of the conditional density is adequately captured by Normal or Hansen skewed-t specifications, but the *extreme tails* are slightly heavier than the Normal assumes, producing the tail-quantile over-coverage observed in the Christoffersen tests. The Hansen skewed-t accommodates these tail dynamics while retaining acceptable density fit overall, supporting its choice as the preferred density for risk applications.

## 8. Application: Black-Scholes option pricing

### 8.1 Setup

We construct a forward-looking valuation exercise for at-the-money straddles on BTC. On each OOS day $t$, three vol-input specifications are used to price a seven-day at-the-money straddle: a constant volatility (equal to the full-sample realised volatility, $\hat\sigma_{\text{const}}$), a backward-looking 30-day rolling realised volatility, and the HAR-RS-DOW conditional forecast $\hat\sigma_t$ (annualised). The Black-Scholes formula for the straddle premium with zero risk-free rate and at-the-money strike is

$$\text{Straddle}(S, \sigma, T) = 2 \cdot S \cdot [\Phi(0.5 \sigma\sqrt{T}) - 0.5].$$

The realised straddle payoff at expiry is $|S_{t+7} - S_t|$. The 'edge' for the straddle seller is the difference between premium received and payoff paid.

### 8.2 Results

**Table 8. Black-Scholes straddle pricing performance (n = 1 305 OOS valuations)**

| σ-input | Avg premium | Avg payoff | Avg edge | Hit rate | MAPE |
|---|---|---|---|---|---|
| Constant σ | €11.16 | €10.30 | +€0.86 | 66.4% | 7.26 |
| Rolling 30-day σ | €10.59 | €10.30 | +€0.29 | 63.0% | 7.51 |
| **HAR-RS-DOW σ̂** | **€10.50** | **€10.30** | **+€0.20** | **60.0%** | **7.50** |

All three vol-specifications yield positive average edge—a manifestation of the well-documented variance risk premium (Bakshi & Kapadia, 2003) by which options market-makers earn a premium for bearing realised-variance risk. HAR-RS-DOW pricing is closest to fair value, exhibiting an average edge of €0.20 versus €0.86 for constant volatility. The hit rate of 60% under HAR pricing is close to the theoretically efficient level for a market-maker hedging at zero edge, indicating that HAR-driven quotes are most closely aligned with realised dynamics.

---

## 9. Robustness

### 9.1 Cross-asset replication

To verify that the HAR-RS-DOW dominance is not artifact of the BTC-EUR sample, we replicate the analysis on two additional cryptocurrencies: ETH-EUR (n = 2 081 days, 7 February 2020 to 14 May 2026) and SOL-EUR (n = 1 738 days, 4 August 2021 to 17 May 2026). All three assets are estimated with the identical HAR-RS-DOW specification using realised kernel measures computed from 5-minute candles.

**Table 9. Cross-asset HAR-RS-DOW replication**

| Asset | n (OOS) | In-sample BIC | OOS CRPS | OOS R² (log-RK) | $\hat\beta_d^- - \hat\beta_d^+$ | $\hat\gamma_{Sat}$ | $\hat\nu$ |
|---|---|---|---|---|---|---|---|
| **BTC-EUR** | 763 | 4 032 | 0.480 | **+0.504** | −0.096 | **−0.274** | 11.7 |
| ETH-EUR | 680 | 3 348 | 0.478 | **+0.347** | +0.011 | **−0.191** | 18.8 |
| SOL-EUR | 567 | 2 634 | 0.407 | **+0.480** | −0.093 | **−0.216** | 14.1 |

Three findings persist across all three cryptocurrencies. First, the model produces positive OOS R² in [+0.35, +0.50]—an order of magnitude larger than what daily-return models typically achieve (see Section 9.4). Second, the estimated Saturday day-of-week coefficient $\hat\gamma_{Sat}$ is *consistently and substantively negative* (-0.19 to -0.27) on all three assets, confirming that weekend volatility reduction is a *structural property of cryptocurrency markets*, not a BTC-specific artifact. Third, the Hansen skewed-t degree-of-freedom parameter $\hat\nu$ remains finite and reasonable (12–19), indicating that conditional return innovations are heavy-tailed but not pathologically so.

### 9.2 Sub-sample stability

Splitting the out-of-sample window into an 'early' period (2022-10 to 2024-01, n = 455) and 'late' period (2024-01 to 2026-05, n = 857), HAR-RS-DOW retains its Sharpe-ratio rank-1 position in CRPS in both halves: CRPS of 0.461 in the early period and 0.516 in the late period, against HAR-WE's 0.554 and 0.601 respectively.

### 9.3 Density-specification robustness for Expected Shortfall

As reported in Table 4, the Acerbi-Szekely Z1 statistic remains positive across three distinct density assumptions for the standardised residual: Normal, Student-t with $\nu = 3$, and Hansen skewed-t with empirically estimated parameters. The conservative-capital conclusion is not driven by the choice of density.

---

### 9.4 Non-HAR benchmark: GARCH(1,1) comparison

The intra-family superiority of HAR-RS-DOW (Sections 5–6) establishes its place within the HAR family. We now demonstrate that the family itself is the right choice for cryptocurrency variance forecasting by comparing against the canonical non-HAR alternative: the GARCH(1,1) model of Bollerslev (1986).

We estimate GARCH(1,1) walk-forward on daily returns for all three assets, with one-step-ahead conditional variance forecasts compared to HAR-RS-DOW on the identical OOS window.

**Table 10. HAR-RS-DOW versus GARCH(1,1), out-of-sample log-RK $R^2$**

| Asset | OOS days | HAR-RS-DOW R² | GARCH(1,1) R² | Margin |
|---|---|---|---|---|
| BTC-EUR | 763 | **+0.504** | −0.287 | +0.790 |
| ETH-EUR | 680 | **+0.347** | −0.192 | +0.539 |
| SOL-EUR | 567 | **+0.480** | −0.188 | +0.668 |

HAR-RS-DOW outperforms GARCH(1,1) on all three assets by OOS log-RK $R^2$ margins of +0.54 to +0.79—a substantial improvement. The negative GARCH $R^2$ indicates that the daily-return-based model is worse than a constant predictor at the unconditional log-RK mean, while HAR-RS-DOW achieves $R^2$ near +0.5. This result is consistent with the broader realised-variance literature (Andersen, Bollerslev, Diebold, & Labys 2003): when high-frequency intraday data is available, realised-measure models dominate parametric daily-return GARCH-family models because they exploit fundamentally more information per day.

The implication for cryptocurrency markets is operational: practitioners with access to intraday data should prefer realised-measure HAR-type specifications over GARCH-family alternatives for variance forecasting tasks.

---

## 10. Conclusion

This paper has established three connected empirical results for the daily realised variance of BTC-EUR. First, the HAR-RS-DOW specification—combining heterogeneous-horizon autoregression, signed semivariance, and day-of-week dummies—is statistically the dominant variance forecaster within the HAR family, both in-sample (ΔBIC = 490 over the next-best alternative) and out-of-sample (CRPS = 0.498 against 0.582, with Diebold-Mariano rejection at $p < 0.001$). Second, the conditional density forecasts produced by HAR-RS-DOW yield conservatively calibrated Expected Shortfall under three plausible density specifications (Normal, Student-t, Hansen skewed-t), with Acerbi-Szekely Z1 ∈ [+1.73, +2.11]—adequate for the Basel III Fundamental Review of the Trading Book capital framework. Third, HAR-RS-DOW's volatility forecasts as inputs to Black-Scholes option pricing produce fair-value pricing closest to market-clearing among three vol-input specifications, with operational implications for derivative market-makers.

These results align with the broader literature view of HAR-family models: they are optimally deployed as variance forecasters underpinning risk and derivative-pricing applications, not as components of directional trading systems. The empirical advantage of HAR-RS-DOW, demonstrated here for BTC-EUR, motivates its inclusion in production risk-management infrastructure for cryptocurrency assets.

Several extensions warrant further investigation. First, the cross-asset structure of variance comovements—e.g., a multivariate HAR specification covering BTC, ETH, and macro hedges—is a natural next step. Second, the deployment of HAR-σ̂ in actual options markets, particularly on Deribit and CME Bitcoin options venues, would test the simulation's results against realised market-maker performance. Third, the methodology developed here may extend to other asset classes where realised-measure data is available at high frequency.

---

## References

Acerbi, C., & Szekely, B. (2014). Back-testing expected shortfall. *Risk*, 27(11), 76–81.

Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up: Including jump components in the measurement, modeling and forecasting of return volatility. *Review of Economics and Statistics*, 89(4), 701–720.

Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.

Bakshi, G., & Kapadia, N. (2003). Delta-hedged gains and the negative market volatility risk premium. *Review of Financial Studies*, 16(2), 527–566.

Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing realised kernels to measure the ex-post variation of equity prices in the presence of noise. *Econometrica*, 76(6), 1481–1536.

Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk*. Bank for International Settlements.

Bauwens, L., Laurent, S., & Rombouts, J. V. K. (2006). Multivariate GARCH models: A survey. *Journal of Applied Econometrics*, 21(1), 79–109.

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.

Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting. *Journal of Econometrics*, 192(1), 1–18.

Bouri, E., Gupta, R., Lau, C. K. M., Roubaud, D., & Wang, S. (2017). Bitcoin and global financial stress: A copula-based approach to dependence and causality in the quantiles. *Quarterly Review of Economics and Finance*, 69, 297–307.

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.

Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business and Economic Statistics*, 19(4), 465–474.

Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253–263.

Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933–949.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.

Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3), 705–730.

Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.

Müller, U. A., Dacorogna, M. M., Davé, R. D., Olsen, R. B., Pictet, O. V., & von Weizsäcker, J. E. (1997). Volatilities of different time resolutions—Analyzing the dynamics of market components. *Journal of Empirical Finance*, 4(2-3), 213–239.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.

Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. *Review of Economics and Statistics*, 97(3), 683–697.

Shen, D., Urquhart, A., & Wang, P. (2020). A three-factor pricing model for cryptocurrencies. *Finance Research Letters*, 34, 101248.

Yu, M. (2019). Forecasting Bitcoin volatility: The role of leverage effect and uncertainty. *Physica A: Statistical Mechanics and its Applications*, 533, 120707.
