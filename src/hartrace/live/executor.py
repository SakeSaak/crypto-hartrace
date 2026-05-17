"""Executor — orchestreert de complete daily cycle.

Per dag:
  1. Forecast σ̂_{tomorrow} via HAR-RS-Q-WE-X
  2. Bepaal target weight w* = min(1, target_daily / σ̂)
  3. Lees huidige BTC-positie (auth API)
  4. Bereken delta_eur (positief = kopen, negatief = verkopen)
  5. Alle safety checks passeren
  6. Plaats limit order (post-only, maker fee)
  7. Log alles, wacht op fill of cancel-na-timeout
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .bitvavo_client import BitvavoClient, OrderRequest, OrderResponse
from .config import TradingConfig
from .forecaster import Forecaster
from .trend_filter import TrendFilter, TrendSignal
from .atr_filter import AtrTrendFilter, AtrTrendSignal
from .safety import SafetyGate

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Volledig record van wat de bot besloot en deed deze cyclus."""
    timestamp: str
    asof_date: str
    sigma_pred: float
    log_rk_pred: float
    target_weight: float
    btc_price_eur: float
    btc_balance: float
    eur_balance: float
    portfolio_eur: float
    current_weight: float
    delta_eur: float
    delta_btc: float
    action: str             # 'buy', 'sell', 'hold', 'blocked', 'dry-logged'
    block_reason: str = ''
    order_id: Optional[str] = None
    order_status: Optional[str] = None
    filled_btc: Optional[float] = None
    filled_eur: Optional[float] = None
    fee_paid: Optional[float] = None
    mode: str = 'unknown'   # 'paper', 'dry-run', 'live'


class Executor:
    """Daily-cycle orchestrator."""
    
    def __init__(self, cfg: TradingConfig, cli_live: bool = False, paper: bool = False):
        self.cfg = cfg
        self.cli_live = cli_live
        self.paper = paper
        self.gate = SafetyGate(cfg)
        project_root = Path(__file__).parent.parent.parent.parent
        if cfg.strategy == 'vol_managed':
            self.forecaster = Forecaster(project_root=project_root)
            self.trend_filter = None
            self.atr_filter = None
        elif cfg.strategy == 'pure_trend':
            self.forecaster = None
            self.trend_filter = TrendFilter(
                project_root=project_root,
                ma_window=cfg.trend_ma_window
            )
            self.atr_filter = None
        elif cfg.strategy == 'atr_trend':
            self.forecaster = None
            self.trend_filter = None
            self.atr_filter = AtrTrendFilter(
                project_root=project_root,
                ma_window=cfg.trend_ma_window,
                atr_window=cfg.atr_window,
                atr_multiplier=cfg.atr_multiplier,
            )
        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy}")
        if not paper:
            self.client = BitvavoClient(cfg.api_key, cfg.api_secret)
        else:
            self.client = None  # paper mode: public-only via separate call
        
        # Determine effective mode
        if paper:
            self.mode = 'paper'
        elif cfg.dry_run or not cli_live or not cfg.live_trading:
            self.mode = 'dry-run'
        else:
            self.mode = 'live'
    
    def fetch_portfolio_state(self) -> tuple:
        """Returns (btc_balance, eur_balance, btc_price)."""
        if self.paper:
            # Paper mode: load state from file or default to zero
            state_path = self.cfg.state_file
            if state_path.exists():
                s = json.loads(state_path.read_text())
                btc = s.get('btc_balance', 0.0)
                eur = s.get('eur_balance', 110.0)  # match thesis €110 backtest
            else:
                btc, eur = 0.0, 110.0
            # Get public price
            client = BitvavoClient(None, None)
            price = client.get_ticker_price(self.cfg.market)
            return btc, eur, price
        
        balances = self.client.get_balance()
        btc = 0.0; eur = 0.0
        base, quote = self.cfg.market.split('-')
        for b in balances:
            if b['symbol'] == base:
                btc = float(b['available']) + float(b.get('inOrder', 0))
            if b['symbol'] == quote:
                eur = float(b['available']) + float(b.get('inOrder', 0))
        price = self.client.get_ticker_price(self.cfg.market)
        return btc, eur, price
    
    def compute_target_weight(self, sigma_pred: float) -> float:
        """w* = min(max_allocation_pct, target_daily / σ̂_pred).
        
        Met max_allocation_pct=0.90 wordt nooit meer dan 90% van portefeuille
        in BTC gezet — 10% blijft als EUR-buffer voor fees, slippage,
        en flexibiliteit bij rebalances.
        """
        target_daily = self.cfg.target_vol_annual / np.sqrt(365)
        w = target_daily / sigma_pred
        # Twee caps: max_allocation_pct (default 0.90) en max_leverage (hard cap 1.0)
        cap = min(self.cfg.max_allocation_pct, self.cfg.max_leverage)
        return float(min(cap, max(0.0, w)))
    
    def compute_delta(self, target_w: float, btc: float, eur: float,
                       price: float) -> tuple:
        """Returns (delta_eur, delta_btc, portfolio_eur, current_w)."""
        portfolio_eur = btc * price + eur
        current_w = (btc * price) / portfolio_eur if portfolio_eur > 0 else 0
        target_eur = target_w * portfolio_eur
        current_eur = btc * price
        delta_eur = target_eur - current_eur
        delta_btc = delta_eur / price
        return delta_eur, delta_btc, portfolio_eur, current_w
    
    def place_order(self, side: str, btc_amount: float,
                      ref_price: float) -> Optional[OrderResponse]:
        """Plaats limit order met offset voor snelle maker-fill."""
        offset = self.cfg.limit_offset_bps * 1e-4
        # buy → bied iets onder mid; sell → vraag iets boven mid
        if side == 'buy':
            limit_price = ref_price * (1 - offset)
        else:
            limit_price = ref_price * (1 + offset)
        
        req = OrderRequest(
            market=self.cfg.market,
            side=side,
            amount=abs(btc_amount),
            price=limit_price,
            post_only=self.cfg.use_post_only,
            operator_id=self.cfg.operator_id,
        )
        logger.info(f"Placing {side} {abs(btc_amount):.8f} {self.cfg.market.split('-')[0]} "
                    f"@ €{limit_price:.2f} (post_only={req.post_only})")
        return self.client.place_limit_order(req)
    
    def _read_state(self) -> dict:
        """Read full state dict (merged across calls)."""
        sp = self.cfg.state_file
        if sp.exists():
            try:
                return json.loads(sp.read_text())
            except Exception:
                return {}
        return {}
    
    def _read_state_atr(self) -> dict:
        """Return ATR-specific state fields only."""
        s = self._read_state()
        stop = s.get('atr_current_stop')
        if stop is None:
            stop = float('-inf')
        return {
            'atr_in_position': bool(s.get('atr_in_position', False)),
            'atr_current_stop': float(stop),
        }
    
    def _merge_state(self, updates: dict):
        """Read-modify-write state file; preserves other fields."""
        s = self._read_state()
        s.update(updates)
        s['updated'] = datetime.now(timezone.utc).isoformat()
        # Float('-inf') is niet JSON-serializable; cast naar string of overslaan
        for k, v in list(s.items()):
            if isinstance(v, float) and not (v == v) or v == float('-inf') or v == float('inf'):
                s[k] = None
        self.cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.state_file.write_text(json.dumps(s, indent=2))
    
    def write_paper_state(self, btc: float, eur: float):
        """Update paper-trading balances (merged)."""
        self._merge_state({'btc_balance': btc, 'eur_balance': eur})
    
    def _persist_atr_state(self):
        """If ATR-strategy heeft pending state, schrijf naar disk."""
        pending = getattr(self, '_pending_atr_state', None)
        if pending is None:
            return
        stop = pending.get('atr_current_stop', float('-inf'))
        # JSON kan geen -inf, cast naar None om "geen stop" te tekenen
        if stop == float('-inf') or stop != stop:  # -inf of NaN
            stop_json = None
        else:
            stop_json = float(stop)
        self._merge_state({
            'atr_in_position': bool(pending.get('atr_in_position', False)),
            'atr_current_stop': stop_json,
        })
        self._pending_atr_state = None
    
    def log_decision(self, decision: TradingDecision):
        """Append decision to daily JSONL log."""
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        log_file = self.cfg.log_dir / f'trades_{date_str}.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(decision)) + '\n')
        logger.info(f"Decision logged: {log_file}")
    
    def _make_decision(self, now, asof_str, sigma_meta, log_meta, target_w,
                         price, btc, eur, portfolio_eur, current_w,
                         delta_eur, delta_btc, action, reason=''):
        d = TradingDecision(
            timestamp=now.isoformat(), asof_date=asof_str,
            sigma_pred=sigma_meta, log_rk_pred=log_meta,
            target_weight=target_w, btc_price_eur=price, btc_balance=btc,
            eur_balance=eur, portfolio_eur=portfolio_eur,
            current_weight=current_w, delta_eur=delta_eur, delta_btc=delta_btc,
            action=action, block_reason=reason, mode=self.mode)
        self.log_decision(d)
        self._persist_atr_state()
        return d
    
    def _empty_decision(self, now, asof, action, reason):
        return TradingDecision(
            timestamp=now.isoformat(), asof_date=asof,
            sigma_pred=0, log_rk_pred=0, target_weight=0,
            btc_price_eur=0, btc_balance=0, eur_balance=0,
            portfolio_eur=0, current_weight=0, delta_eur=0,
            delta_btc=0, action=action, block_reason=reason,
            mode=self.mode)
    
    def _compute_signal(self):
        """Returns (target_w, asof_date_str, sigma_pred, log_rk_pred, sanity_ok, block_reason).
        
        Branches op strategy.
        """
        if self.cfg.strategy == 'vol_managed':
            forecast = self.forecaster.forecast()
            logger.info(f"Forecast: σ̂={forecast.sigma_pred_daily:.5f} "
                        f"(log_RK={forecast.log_rk_pred:.4f}, "
                        f"asof={forecast.asof_date.date()}, "
                        f"fit_age={forecast.fit_age_days}d)")
            fs = self.gate.forecast_in_range(forecast.sigma_pred_daily)
            if not fs:
                return 0, forecast.asof_date.isoformat(), forecast.sigma_pred_daily, \
                       forecast.log_rk_pred, False, fs.reason
            target_w = self.compute_target_weight(forecast.sigma_pred_daily)
            return target_w, forecast.asof_date.isoformat(), \
                   forecast.sigma_pred_daily, forecast.log_rk_pred, True, ''
        
        elif self.cfg.strategy == 'pure_trend':
            signal = self.trend_filter.compute()
            target_w = self.cfg.max_allocation_pct if signal.in_trend else 0.0
            logger.info(f"Pure trend strategy: target_w={target_w:.2f} "
                        f"({'IN-TREND' if signal.in_trend else 'OUT-OF-TREND'}, "
                        f"deviation={signal.deviation_pct*100:+.2f}%)")
            # sigma_pred field hergebruikt voor MA deviation, log_rk_pred voor MA value
            return target_w, signal.asof_date.isoformat(), \
                   signal.deviation_pct, signal.ma_value, True, ''
        
        elif self.cfg.strategy == 'atr_trend':
            state = self._read_state_atr()
            prior_in_pos = state['atr_in_position']
            prior_stop = state['atr_current_stop']
            sig = self.atr_filter.compute(prior_in_pos, prior_stop)
            target_w = self.cfg.max_allocation_pct if sig.target_w > 0 else 0.0
            # Save new state to be persisted at end of run
            self._pending_atr_state = {
                'atr_in_position': sig.in_position_after,
                'atr_current_stop': sig.new_stop,
            }
            logger.info(
                f"ATR-stop signal: action={sig.action}, price=€{sig.current_price:.2f}, "
                f"MA{sig.ma_window}=€{sig.ma_value:.2f}, ATR={sig.atr_value:.2f}, "
                f"prior_pos={prior_in_pos}, new_stop=€{sig.new_stop:.2f}, "
                f"target_w={target_w:.2f}"
            )
            # sigma_pred field hergebruikt voor ATR-value, log_rk_pred voor stop-level
            return target_w, sig.asof_date.isoformat(), \
                   sig.atr_value, sig.new_stop, True, ''
        
        else:
            raise ValueError(f"Unknown strategy: {self.cfg.strategy}")
    
    def run_once(self) -> TradingDecision:
        """Eén complete trading cycle. Idempotent — kan herhaaldelijk."""
        now = datetime.now(timezone.utc)
        logger.info(f"=== Run start {now.isoformat()} mode={self.mode} "
                    f"strategy={self.cfg.strategy} ===")
        
        # 0. Kill switch check (ALTIJD eerst)
        ks = self.gate.kill_switch_active()
        if not ks:
            logger.error(f"BLOCKED: {ks.reason}")
            return self._empty_decision(now, '', 'blocked', ks.reason)
        
        # 0b. Skip-weekend filter (H10: HAR-RS-DOW γ_saturday = -0.281)
        # Weekend dagen hebben systematisch lagere vol → wijdere spreads → 
        # duurdere executie. Skip-weekend levert +0.06 Sharpe + 25% minder fees.
        if self.cfg.skip_weekend and now.weekday() in (5, 6):
            dow_name = now.strftime('%A')
            logger.info(f"SKIP WEEKEND: vandaag is {dow_name} "
                        f"(DOW={now.weekday()}). HAR-RS-DOW DOW γ-coefficient "
                        f"toont systematisch lage vol → wijdere spreads. Geen trade.")
            return self._empty_decision(
                now, '', 'skip_weekend',
                f'weekend ({dow_name}, low-vol DOW per HAR-RS-DOW)')
        
        # 1+2. Compute signal (strategy-specific)
        target_w, asof_str, sigma_meta, log_meta, ok, reason = self._compute_signal()
        if not ok:
            logger.error(f"BLOCKED signal: {reason}")
            return self._empty_decision(now, asof_str, 'blocked', reason)
        
        # 3. Portfolio state
        btc, eur, price = self.fetch_portfolio_state()
        delta_eur, delta_btc, portfolio_eur, current_w = self.compute_delta(
            target_w, btc, eur, price)
        target_eur = target_w * portfolio_eur
        
        logger.info(f"Portfolio: BTC={btc:.8f} (€{btc*price:.2f}), "
                    f"EUR={eur:.2f}, total €{portfolio_eur:.2f}")
        logger.info(f"Current w={current_w:.3f}, target w={target_w:.3f}, "
                    f"delta €{delta_eur:.2f} ({delta_btc:+.8f} BTC)")
        
        # 4. Rebalance trigger check (strategy-specific)
        if self.cfg.strategy == 'pure_trend':
            # Pure trend: rebalance alleen bij |Δw| > deadband
            if abs(target_w - current_w) <= self.cfg.trend_deadband:
                logger.info(f"HOLD: |Δw|={abs(target_w - current_w):.3f} "
                            f"binnen deadband {self.cfg.trend_deadband:.2f}, geen actie")
                return self._make_decision(
                    now, asof_str, sigma_meta, log_meta, target_w, price, btc, eur,
                    portfolio_eur, current_w, delta_eur, delta_btc, 'hold',
                    'binnen trend-deadband (geen significante drift)')
        
        # 5. Safety stack: ALLE checks moeten passeren
        checks = [
            self.gate.position_cap(target_eur),
            self.gate.turnover_cap(delta_eur, portfolio_eur),
            self.gate.min_trade_size(delta_eur),
        ]
        gate_result = self.gate.all_checks(*checks)
        
        decision = TradingDecision(
            timestamp=now.isoformat(), asof_date=asof_str,
            sigma_pred=sigma_meta, log_rk_pred=log_meta,
            target_weight=target_w, btc_price_eur=price, btc_balance=btc,
            eur_balance=eur, portfolio_eur=portfolio_eur, current_weight=current_w,
            delta_eur=delta_eur, delta_btc=delta_btc, action='', mode=self.mode)
        
        if not gate_result:
            decision.action = 'hold'
            decision.block_reason = gate_result.reason
            logger.info(f"HOLD: {gate_result.reason}")
            self._persist_atr_state()
            self.log_decision(decision)
            return decision
        
        side = 'buy' if delta_btc > 0 else 'sell'
        
        # 6. Mode-specific execution
        if self.mode == 'paper':
            # Simulate fill at current price minus fee
            fee_rate = 0.0015  # maker
            fee = abs(delta_eur) * fee_rate
            if side == 'buy':
                new_btc = btc + delta_btc
                new_eur = eur - delta_eur - fee
            else:
                new_btc = btc + delta_btc  # delta_btc is negative
                new_eur = eur - delta_eur - fee  # delta_eur is negative
            self.write_paper_state(new_btc, new_eur)
            decision.action = 'paper-filled'
            decision.filled_btc = abs(delta_btc)
            decision.filled_eur = abs(delta_eur)
            decision.fee_paid = fee
            decision.order_status = 'filled (simulated)'
            logger.info(f"PAPER fill: {side} {abs(delta_btc):.8f} BTC, fee €{fee:.4f}")
        
        elif self.mode == 'dry-run':
            decision.action = 'dry-logged'
            logger.info(f"DRY-RUN: zou {side} {abs(delta_btc):.8f} BTC plaatsen "
                        f"voor ~€{abs(delta_eur):.2f}")
        
        else:  # live
            # Triple-check live authorization
            live_check = self.gate.live_trading_authorized(self.cli_live)
            if not live_check:
                decision.action = 'blocked'
                decision.block_reason = live_check.reason
                logger.error(f"BLOCKED live: {live_check.reason}")
                self._persist_atr_state()
                self.log_decision(decision)
                return decision
            
            try:
                order_resp = self.place_order(side, abs(delta_btc), price)
                decision.action = side
                decision.order_id = order_resp.order_id
                logger.info(f"Order placed: id={order_resp.order_id}, "
                            f"initial status={order_resp.status}")
                
                # Wait for fill of cancel-na-timeout
                final = self.client.wait_for_fill_or_cancel(
                    market=self.cfg.market,
                    order_id=order_resp.order_id,
                    timeout_sec=self.cfg.order_timeout_sec,
                    poll_interval_sec=30,
                )
                decision.order_status = final.get('status', 'unknown')
                decision.filled_btc = float(final.get('filledAmount', 0))
                decision.filled_eur = float(final.get('filledAmountQuote', 0))
                decision.fee_paid = float(final.get('feePaid', 0))
                
                if decision.filled_btc == 0:
                    logger.warning(f"Order eindigde zonder fill (status={decision.order_status}). "
                                   f"Probeer morgen opnieuw, of switch tijdelijk naar taker.")
                else:
                    logger.info(f"Order eindstatus: {decision.order_status}, "
                                f"filled {decision.filled_btc:.8f} BTC "
                                f"voor €{decision.filled_eur:.2f}, "
                                f"fee €{decision.fee_paid:.4f} {final.get('feeCurrency', '')}")
            except Exception as e:
                decision.action = 'error'
                decision.block_reason = str(e)
                logger.error(f"Order placement/tracking failed: {e}")
        
        self._persist_atr_state()
        
        self.log_decision(decision)
        return decision
