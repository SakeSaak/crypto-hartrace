"""Config — env-loader, three-layer safety, kill switches."""
from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingConfig:
    """Immutable runtime configuration."""
    # Credentials (loaded from env, never logged)
    api_key: Optional[str]
    api_secret: Optional[str]
    operator_id: Optional[str]
    
    # Safety flags
    live_trading: bool      # env-level switch, default False
    dry_run: bool           # if True: log decisions, don't place orders
    
    # Strategy selection
    strategy: str = 'pure_trend'  # H8 winner — empirisch beste over alle metrics , 'pure_trend' (G5), 'vol_managed' (HAR)
    
    # Strategy parameters - shared
    market: str = 'BTC-EUR'
    max_leverage: float = 1.0  # Bitvavo spot is long-only, hard cap
    max_allocation_pct: float = 0.90  # max % van portfolio dat in BTC mag (rest = EUR buffer)
    
    # Strategy parameters - vol-managed (alleen relevant als strategy='vol_managed')
    target_vol_annual: float = 0.35
    
    # Strategy parameters - pure trend (alleen relevant als strategy='pure_trend')
    trend_ma_window: int = 50              # 50d MA (G5: plateau 40-80d, conventioneel 50)
    trend_deadband: float = 0.05           # rebalance alleen bij |Δw| > 5%
    
    # Strategy parameters - atr_trend (alleen relevant als strategy='atr_trend')
    atr_window: int = 14                   # ATR-periode (Wilder 1978 conventie)
    atr_multiplier: float = 3.0            # stop-afstand (H3-multiplier sweep: piek bij 3.0)
    
    # Hard limits (code-level safety)
    max_position_eur: float = 1000.0       # absolute cap on BTC exposure
    max_daily_turnover_pct: float = 0.30   # refuse rebalance > 30% of portfolio
    min_trade_eur: float = 5.0             # Bitvavo minimum
    forecast_sigma_min: float = 0.005      # σ̂ < 0.5% daily = anomalous low
    forecast_sigma_max: float = 0.30       # σ̂ > 30% daily = anomalous high
    
    # Order placement
    order_timeout_sec: int = 300           # cancel limit order na 5 min als niet vol
    use_post_only: bool = True             # only maker fills (15 bps fee)
    limit_offset_bps: float = 5.0          # place at mid ± 5 bps for fast fill
    
    # Operational
    kill_switch_file: Path = field(default_factory=lambda: Path.home() / 'STOP_TRADING')
    state_file: Path = field(
        default_factory=lambda: Path.home() / 'Desktop/crypto_hartrace/state/trader_state.json'
    )
    log_dir: Path = field(
        default_factory=lambda: Path.home() / 'Desktop/crypto_hartrace/logs'
    )
    
    def __post_init__(self):
        # Ensure log_dir and state_file dir exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path) -> dict:
    """Parse a .env file into dict without exposing values in logs."""
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    
    if env_path.stat().st_mode & 0o077 != 0:
        logger.warning(
            f"⚠️  {env_path} has loose permissions. "
            f"Run: chmod 600 {env_path}"
        )
    
    env = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_config(
    env_path: Path = Path.home() / 'W8W.env',
    require_keys: bool = True,
) -> TradingConfig:
    """Load and validate configuration.
    
    Three-layer safety check:
      1. DRY_RUN defaults to True if not explicitly set to 'false'
      2. LIVE_TRADING defaults to False unless explicitly set to 'true'
      3. Both flags must be aligned for live trading to occur
    """
    env = load_env_file(env_path)
    
    api_key = env.get('API_KEY')
    api_secret = env.get('API_SECRET')
    operator_id = env.get('BITVAVO_OPERATOR_ID')
    
    if require_keys and not (api_key and api_secret):
        raise ValueError(
            f"API_KEY and/or API_SECRET missing in {env_path}. "
            f"Required for non-paper modes."
        )
    
    # CRITICAL: defaults err on side of safety
    dry_run_raw = env.get('DRY_RUN', 'true').lower()
    live_raw = env.get('LIVE_TRADING', 'false').lower()
    
    dry_run = dry_run_raw != 'false'    # any non-'false' = dry run = SAFE
    live = live_raw == 'true'            # explicit 'true' required
    
    if live and dry_run:
        logger.warning(
            "Both LIVE_TRADING=true and DRY_RUN=true. "
            "DRY_RUN wins — no real orders will be placed."
        )
    
    # Strategy selection
    strategy = env.get('STRATEGY', 'pure_trend').lower()
    if strategy not in ('atr_trend', 'pure_trend', 'vol_managed'):
        raise ValueError(
            f"Invalid STRATEGY={strategy}. Use 'atr_trend', 'pure_trend' of 'vol_managed'."
        )
    
    # Strategy-specific turnover cap:
    # - atr_trend / pure_trend: 1.0 (signal flips 0→90% of 90%→0% zijn legitiem)
    # - vol_managed: 0.30 (kleine vol-gerelateerde aanpassingen verwacht)
    default_turn = 1.0 if strategy in ('atr_trend', 'pure_trend') else 0.30
    max_turn = float(env.get('MAX_DAILY_TURNOVER_PCT', default_turn))
    
    # Max position (in EUR) cap — kan ook via env override
    max_pos = float(env.get('MAX_POSITION_EUR', 1000.0))
    
    cfg = TradingConfig(
        api_key=api_key, api_secret=api_secret, operator_id=operator_id,
        live_trading=live, dry_run=dry_run,
        strategy=strategy,
        max_daily_turnover_pct=max_turn,
        max_position_eur=max_pos,
    )
    
    # Log redacted summary (never log keys)
    logger.info(f"Config loaded: market={cfg.market}, "
                f"target_vol={cfg.target_vol_annual:.0%}, "
                f"dry_run={cfg.dry_run}, live_trading={cfg.live_trading}, "
                f"max_pos_eur={cfg.max_position_eur}, "
                f"api_key=***{(api_key or '')[-4:] if api_key else 'none'}")
    return cfg
