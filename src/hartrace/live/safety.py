"""Safety — kill switch, position caps, sanity checks.

EVERY pre-trade decision passes through SafetyGate. Multiple independent
checks; any single fail aborts the trade.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from .config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    passed: bool
    reason: str = ''
    
    def __bool__(self):
        return self.passed


class SafetyGate:
    """All pre-trade safety checks. Fail-closed: if anything ambiguous → block."""
    
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
    
    def kill_switch_active(self) -> SafetyCheckResult:
        """Operator can stop all trading by `touch ~/STOP_TRADING`."""
        if self.cfg.kill_switch_file.exists():
            return SafetyCheckResult(
                passed=False,
                reason=f"Kill switch active: {self.cfg.kill_switch_file}. "
                       f"Remove file to resume."
            )
        return SafetyCheckResult(passed=True)
    
    def forecast_in_range(self, sigma_pred: float) -> SafetyCheckResult:
        """Anomalous forecasts (likely bugs) → block."""
        if sigma_pred < self.cfg.forecast_sigma_min:
            return SafetyCheckResult(
                passed=False,
                reason=f"σ̂={sigma_pred:.5f} below floor {self.cfg.forecast_sigma_min}. "
                       f"Possible model failure or stale data."
            )
        if sigma_pred > self.cfg.forecast_sigma_max:
            return SafetyCheckResult(
                passed=False,
                reason=f"σ̂={sigma_pred:.5f} above ceiling {self.cfg.forecast_sigma_max}. "
                       f"Possible flash event; manual review required."
            )
        return SafetyCheckResult(passed=True)
    
    def position_cap(self, target_eur: float) -> SafetyCheckResult:
        """Hard cap on total BTC exposure."""
        if target_eur > self.cfg.max_position_eur:
            return SafetyCheckResult(
                passed=False,
                reason=f"Target position €{target_eur:.2f} > cap "
                       f"€{self.cfg.max_position_eur:.2f}. Refused."
            )
        return SafetyCheckResult(passed=True)
    
    def turnover_cap(self, delta_eur: float, portfolio_eur: float) -> SafetyCheckResult:
        """Reject trades that rotate too much of the portfolio in one day."""
        if portfolio_eur <= 0:
            return SafetyCheckResult(passed=True)  # nothing to rebalance
        pct = abs(delta_eur) / portfolio_eur
        if pct > self.cfg.max_daily_turnover_pct:
            return SafetyCheckResult(
                passed=False,
                reason=f"Trade |€{delta_eur:.2f}| = {pct:.1%} of portfolio, "
                       f"exceeds cap {self.cfg.max_daily_turnover_pct:.0%}. "
                       f"Likely state drift; manual review required."
            )
        return SafetyCheckResult(passed=True)
    
    def min_trade_size(self, delta_eur: float) -> SafetyCheckResult:
        """Skip sub-minimum trades (Bitvavo €5 floor)."""
        if abs(delta_eur) < self.cfg.min_trade_eur:
            return SafetyCheckResult(
                passed=False,
                reason=f"|€{delta_eur:.2f}| below Bitvavo minimum "
                       f"€{self.cfg.min_trade_eur}. Skipping."
            )
        return SafetyCheckResult(passed=True)
    
    def live_trading_authorized(self, cli_live_flag: bool) -> SafetyCheckResult:
        """All three independent layers must agree for live trading."""
        env_authorized = self.cfg.live_trading
        env_dry_safe = self.cfg.dry_run
        
        if not env_authorized:
            return SafetyCheckResult(
                passed=False,
                reason="LIVE_TRADING != 'true' in env."
            )
        if env_dry_safe:
            return SafetyCheckResult(
                passed=False,
                reason="DRY_RUN != 'false' in env (default-safe). "
                       "Explicitly set DRY_RUN=false in W8W.env to enable."
            )
        if not cli_live_flag:
            return SafetyCheckResult(
                passed=False,
                reason="--live flag not provided at runtime. "
                       "Bot will not place real orders."
            )
        return SafetyCheckResult(passed=True)
    
    def all_checks(self, *checks: SafetyCheckResult) -> SafetyCheckResult:
        """Combine multiple checks. ALL must pass."""
        for c in checks:
            if not c.passed:
                return c
        return SafetyCheckResult(passed=True)
