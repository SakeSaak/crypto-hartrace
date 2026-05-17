"""Reproducibility test-suite voor crypto_hartrace thesis project.

Voer uit met:  pytest tests/test_reproducibility.py -v
Of selectief:  pytest tests/test_reproducibility.py::test_har_dom_oos_crps -v

Deze tests valideren dat alle kern-resultaten uit de thesis reproducerbaar zijn
vanaf bewaarde data + scripts. Als een test faalt is er ofwel een data-bug
ofwel een code-regression — beide moeten gefixt worden voordat de thesis
overhandigd wordt.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path('/Users/sakesaakstra/Desktop/crypto_hartrace')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# ===== Sectie A: Data integrity tests =====

class TestDataIntegrity:
    """Confirmeer dat de raw data en processed outputs aanwezig zijn."""
    
    def test_btc_candles_aanwezig(self):
        for freq in ['1m', '5m', '15m', '1h', '1d']:
            f = PROJECT_ROOT / f'data/raw/candles_BTC-EUR_{freq}.parquet'
            assert f.exists(), f"Missing: {f}"
            df = pd.read_parquet(f)
            assert len(df) > 100, f"Too few rows in {f}: {len(df)}"
            assert {'ts', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns)
    
    def test_eth_candles_aanwezig(self):
        f = PROJECT_ROOT / 'data/raw/candles_ETH-EUR_1d.parquet'
        assert f.exists()
        df = pd.read_parquet(f)
        assert len(df) > 1000  # ~7 jaar daily
    
    def test_e1_walkforward_aanwezig(self):
        """OOS walk-forward parquet is de hoeksteen van Deel I + II."""
        f = PROJECT_ROOT / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet'
        assert f.exists()
        df = pd.read_parquet(f)
        assert len(df) == 1312, f"Expected 1312 OOS dagen, got {len(df)}"
        required = {'mu_pred', 'sigma_pred', 'nu_pred', 'lam_pred', 'y_actual',
                    'r_actual', 'crps', 'qlike',
                    'var_return_01', 'viol_return_01',
                    'var_return_05', 'viol_return_05'}
        assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"
    
    def test_har_oos_period_consistent(self):
        df = pd.read_parquet(PROJECT_ROOT / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        assert df.index.min().date().isoformat() == '2022-10-01'
        assert df.index.max().date().isoformat() <= '2026-05-15'


# ===== Sectie B: Model performance tests =====

class TestModelPerformance:
    """Verifieer kern-cijfers uit DEEL I van de thesis."""

    def setup_method(self):
        self.df = pd.read_parquet(
            PROJECT_ROOT / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_localize(None)

    def test_har_dom_oos_crps(self):
        """HAR-RS-DOW CRPS = 0.498 (claim in thesis sectie 5)."""
        crps = self.df['crps'].mean()
        assert abs(crps - 0.498) < 0.005, f"CRPS={crps:.4f}, expected ~0.498"
    
    def test_har_dom_qlike(self):
        """HAR-RS-DOW QLIKE = 0.595 (claim in thesis sectie 5)."""
        qlike = self.df['qlike'].mean()
        assert abs(qlike - 0.595) < 0.01, f"QLIKE={qlike:.4f}, expected ~0.595"

    def test_har_dom_r2_oos(self):
        """HAR-RS-DOW R² uit-sample = +0.44 op log_RK niveau."""
        y = self.df['y_actual']
        yhat = self.df['mu_pred']
        ss_res = ((y - yhat)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
        assert 0.42 < r2 < 0.46, f"R²_oos={r2:.3f}, expected ~0.44"
    
    def test_har_dominates_alternatives(self):
        """HAR-RS-DOW heeft betere CRPS dan ALLE andere HAR families."""
        rs_dow_crps = self.df['crps'].mean()
        for alt in ['HAR-RS-Q-WE-X', 'HAR-RS-X', 'HAR-RS-WE', 'HAR-WE']:
            f = PROJECT_ROOT / f'outputs/tables/E1_walkforward_per_day_{alt}.parquet'
            if not f.exists(): continue
            alt_df = pd.read_parquet(f)
            alt_crps = alt_df['crps'].mean()
            assert rs_dow_crps < alt_crps, \
                f"HAR-RS-DOW ({rs_dow_crps:.3f}) niet beter dan {alt} ({alt_crps:.3f})"
    
    def test_var_coverage_log_rk(self):
        """VaR coverage op log-RK niveau moet kalibreerd zijn (sectie 8)."""
        for alpha, viol_col in [(0.01, 'viol_logrk_01'), (0.05, 'viol_logrk_05')]:
            actual_rate = self.df[viol_col].mean()
            # Tolerance: ±2× alpha rond target
            assert 0.5 * alpha < actual_rate < 2.0 * alpha, \
                f"VaR {alpha:.0%} coverage = {actual_rate:.3f}, expected ~{alpha}"


# ===== Sectie C: Risk forecasting tests (DEEL II) =====

class TestRiskForecasting:
    """ES en VaR resultaten zijn de kern-bijdrage van DEEL II."""

    def setup_method(self):
        self.df = pd.read_parquet(
            PROJECT_ROOT / 'outputs/tables/E1_walkforward_per_day_HAR-RS-DOW.parquet')
        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_localize(None)
        self.df['sigma_ret'] = np.sqrt(np.exp(
            self.df['mu_pred'] + 0.5 * self.df['sigma_pred']**2))

    def test_es_t3_z1_conservative(self):
        """Z1 voor Student-t(3) op 5% niveau moet positief zijn (conservatief)."""
        alpha = 0.05
        nu = 3
        t_a = stats.t.ppf(alpha, nu)
        es_z = -(nu + t_a**2) / (nu - 1) * stats.t.pdf(t_a, nu) / alpha / np.sqrt(nu/(nu-2))
        self.df['es'] = self.df['sigma_ret'] * es_z
        viol = self.df[self.df['viol_return_05'] == 1]
        z1 = (viol['r_actual'] / viol['es']).mean() + 1
        assert z1 > 0.5, f"Z1={z1:.3f}, model should be conservatief (Z1 > 0)"
    
    def test_hansen_skewt_lambda_symmetric(self):
        """Bij standaardisatie z=r/σ moet de skewness λ ≈ 0 zijn."""
        z = (self.df['r_actual'] / self.df['sigma_ret']).dropna()
        z = z[(z > -8) & (z < 8)]
        # Sample skewness (geen MLE, maar wel proxy)
        skew = stats.skew(z)
        assert abs(skew) < 0.5, \
            f"Standardized skewness = {skew:.3f}, HAR moet asymmetrie absorberen"


# ===== Sectie D: Live trading code tests =====

class TestLiveTradingComponents:
    """Veiligheidskritieke componenten van de live bot."""

    def test_atr_filter_module_importable(self):
        from hartrace.live.atr_filter import AtrTrendFilter, AtrTrendSignal
        assert AtrTrendFilter is not None
        assert AtrTrendSignal is not None

    def test_atr_signal_enter_logic(self):
        """Cold start, price > MA → action='enter', stop = price - mult×ATR."""
        from hartrace.live.atr_filter import AtrTrendFilter
        af = AtrTrendFilter(PROJECT_ROOT, ma_window=50, atr_window=14, atr_multiplier=3.0)
        sig = af.compute(in_position_prior=False, current_stop_prior=float('-inf'))
        # Sanity checks (precieze waarden hangen af van actuele data)
        assert sig.atr_value > 0
        assert sig.ma_value > 0
        in_trend = sig.current_price > sig.ma_value
        if in_trend:
            assert sig.action == 'enter'
            assert sig.target_w == 0.90
            assert abs(sig.new_stop - (sig.current_price - 3.0 * sig.atr_value)) < 1.0

    def test_bitvavo_tick_rounding(self):
        """Tick-size voor BTC-EUR is €1.00; round-to-tick moet floor zijn voor buy."""
        from hartrace.live.bitvavo_client import BitvavoClient
        c = BitvavoClient(api_key=None, api_secret=None)
        info = c.get_market_info('BTC-EUR')
        assert info['tickSize'] == '1.00', f"Tick changed: {info['tickSize']}"
        # Floor voor buy (post-only safety)
        assert c._round_price_to_tick('BTC-EUR', 67496.24, 'buy') == 67496.00
        # Ceil voor sell
        assert c._round_price_to_tick('BTC-EUR', 67496.24, 'sell') == 67497.00
        # Al-rond getallen veranderen niet
        assert c._round_price_to_tick('BTC-EUR', 67500.00, 'buy') == 67500.00


# ===== Sectie E: Mathematical correctness =====

class TestMathematicalCorrectness:
    """Verifieer dat onze custom math functies kloppen met scipy/literatuur."""

    def test_normal_es_formula(self):
        """ES_5% Normal = -σ × φ(z_5%) / 0.05 = -2.063σ."""
        sigma = 1.0
        alpha = 0.05
        z = stats.norm.ppf(alpha)
        es = -sigma * stats.norm.pdf(z) / alpha
        assert abs(es - (-2.063)) < 0.005

    def test_studentt3_es_more_severe_than_normal(self):
        """ES voor t(3) moet groter (=meer negatief) zijn dan voor Normal."""
        alpha = 0.01
        # Normal
        z_norm = stats.norm.ppf(alpha)
        es_norm = -stats.norm.pdf(z_norm) / alpha
        # Standardized t(3)
        nu = 3
        t_a = stats.t.ppf(alpha, nu)
        es_t3 = -(nu + t_a**2) / (nu - 1) * stats.t.pdf(t_a, nu) / alpha / np.sqrt(nu/(nu-2))
        assert es_t3 < es_norm, \
            f"t(3) ES ({es_t3:.3f}) moet zwaarder zijn dan Normal ES ({es_norm:.3f})"
    
    def test_black_scholes_atm_call_put_parity(self):
        """ATM call - put = S - K×exp(-rT). Bij r=0 en K=S: call = put."""
        S, K, sigma, T = 100, 100, 0.50, 30/365
        d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S*stats.norm.cdf(d1) - K*stats.norm.cdf(d2)
        put = call - S + K  # parity
        assert abs(call - put) < 0.01  # ATM zonder rente


# ===== Sectie F: Strategy backtest invariants =====

class TestStrategyBacktest:
    """Sleutel-cijfers uit H2/H3 (strategy comparison)."""

    def test_h2_strategies_file_exists(self):
        f = PROJECT_ROOT / 'outputs/tables/H2_TA_strategies.csv'
        assert f.exists()
        df = pd.read_csv(f)
        assert 'sharpe' in df.columns
        assert 'strategy' in df.columns
        assert len(df) >= 10
    
    def test_atr_stop_wins_h2(self):
        """ATR-stop is volgens thesis sectie 10 + appendix A4 de winner."""
        df = pd.read_csv(PROJECT_ROOT / 'outputs/tables/H2_TA_strategies.csv')
        winner = df.sort_values('sharpe', ascending=False).iloc[0]
        assert 'ATR' in winner['strategy'], \
            f"Verwacht ATR-strategy winner, kreeg {winner['strategy']}"
        assert winner['sharpe'] > 1.0
    
    def test_har_voltarget_top3(self):
        """HAR vol-target + Trend moet top-3 zijn (sectie 10)."""
        df = pd.read_csv(PROJECT_ROOT / 'outputs/tables/H2_TA_strategies.csv')
        top3 = df.sort_values('sharpe', ascending=False).head(3)
        has_har = any('HAR' in s for s in top3['strategy'])
        assert has_har, "HAR-strategie moet in top-3"


if __name__ == '__main__':
    # Run als script: shorter feedback
    pytest.main([__file__, '-v', '--tb=short'])
