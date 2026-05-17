"""Bitvavo REST API client met HMAC-SHA256 auth.

Public endpoints (no auth):
  - get_candles, get_ticker_price
Private endpoints (auth required):
  - get_balance, get_open_orders
  - place_limit_order, cancel_order

Rate limit: Bitvavo geeft 1000 weight/minuut. Elke call kost weight,
zie 'Bitvavo-Ratelimit-Remaining' header. We tracken dat.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = 'https://api.bitvavo.com/v2'


@dataclass
class OrderRequest:
    market: str
    side: str           # 'buy' or 'sell'
    amount: float       # base asset (BTC)
    price: float        # in quote (EUR)
    post_only: bool = True
    operator_id: Optional[str] = None


@dataclass
class OrderResponse:
    order_id: str
    status: str
    filled_amount: float
    filled_amount_quote: float
    fee_paid: float
    fee_currency: str
    raw: dict


class BitvavoClient:
    """Thin REST wrapper. Throws on HTTP/API errors."""
    
    def __init__(self, api_key: Optional[str], api_secret: Optional[str],
                  access_window_ms: int = 10_000):
        self.api_key = api_key
        self.api_secret = (api_secret or '').encode()
        self.access_window = access_window_ms
        self.session = requests.Session()
        self.rate_limit_remaining = 1000  # updated from response headers
        self._market_cache: dict = {}      # cache voor /v2/markets info (tick-size etc.)
    
    def _sign(self, method: str, path: str, body: str = '') -> dict:
        """Build authentication headers."""
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API credentials required for private endpoints.")
        ts = str(int(time.time() * 1000))
        prehash = ts + method + '/v2' + path + body
        signature = hmac.new(self.api_secret, prehash.encode(), hashlib.sha256).hexdigest()
        return {
            'Bitvavo-Access-Key': self.api_key,
            'Bitvavo-Access-Signature': signature,
            'Bitvavo-Access-Timestamp': ts,
            'Bitvavo-Access-Window': str(self.access_window),
            'Content-Type': 'application/json',
        }
    
    def _request(self, method: str, path: str, params: dict = None,
                   body: dict = None, auth: bool = False) -> dict:
        url = BASE_URL + path
        body_str = json.dumps(body, separators=(',', ':')) if body else ''
        
        # CRITICAL: Bitvavo HMAC vereist dat de signature-path de query-string
        # bevat (deterministisch geserializeerd, geen URL-encoding van veilige
        # tekens). De Bitvavo officiële SDK doet dit ook.
        if params:
            from urllib.parse import urlencode
            qs = urlencode(params, doseq=True)
            sign_path = f"{path}?{qs}"
        else:
            sign_path = path
        
        headers = self._sign(method, sign_path, body_str) if auth else {}
        
        try:
            r = self.session.request(
                method, url, params=params,
                data=body_str if body else None,
                headers=headers, timeout=30
            )
        except requests.RequestException as e:
            logger.error(f"Network error {method} {path}: {e}")
            raise
        
        # Track rate limit
        rem = r.headers.get('Bitvavo-Ratelimit-Remaining')
        if rem:
            self.rate_limit_remaining = int(rem)
            if self.rate_limit_remaining < 50:
                logger.warning(f"Rate limit low: {self.rate_limit_remaining} remaining")
        
        if r.status_code >= 400:
            logger.error(f"Bitvavo {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
        
        return r.json()
    
    # -------- Public endpoints --------
    
    def get_candles(self, market: str, interval: str = '5m', limit: int = 1440):
        """Fetch OHLCV candles. interval in {1m,5m,15m,30m,1h,2h,4h,1d}."""
        return self._request('GET', f'/{market}/candles',
                              params={'interval': interval, 'limit': limit})
    
    def get_ticker_price(self, market: str) -> float:
        """Latest mid price."""
        r = self._request('GET', '/ticker/price', params={'market': market})
        return float(r['price'])
    
    def get_book(self, market: str, depth: int = 10) -> dict:
        """Order book top N levels."""
        return self._request('GET', f'/{market}/book', params={'depth': depth})
    
    # -------- Private endpoints --------
    
    def get_balance(self, symbol: Optional[str] = None) -> list:
        """List balances. symbol e.g. 'BTC' or 'EUR' for filter."""
        params = {'symbol': symbol} if symbol else None
        return self._request('GET', '/balance', params=params, auth=True)
    
    def get_open_orders(self, market: Optional[str] = None) -> list:
        params = {'market': market} if market else None
        return self._request('GET', '/ordersOpen', params=params, auth=True)
    
    def get_market_info(self, market: str) -> dict:
        """Haal en cache market-info (tick-size, decimalen, min-order, etc)."""
        if market in self._market_cache:
            return self._market_cache[market]
        # Public endpoint, geen auth
        info_list = self._request('GET', '/markets', auth=False)
        for m in info_list:
            self._market_cache[m['market']] = m
        if market not in self._market_cache:
            raise ValueError(f"Market {market} niet gevonden bij Bitvavo")
        return self._market_cache[market]
    
    def _round_price_to_tick(self, market: str, price: float, side: str) -> float:
        """Rond limit price af naar de exchange tick-size.
        
        - BUY post-only: floor (we willen ONDER huidige ask, zodat order maker is)
        - SELL post-only: ceil (we willen BOVEN huidige bid, zodat order maker is)
        """
        info = self.get_market_info(market)
        tick = float(info.get('tickSize', '0.01'))
        import math
        if side == 'buy':
            return math.floor(price / tick) * tick
        else:  # sell
            return math.ceil(price / tick) * tick
    
    def place_limit_order(self, req: OrderRequest) -> OrderResponse:
        """Plaats limit order (default post-only voor maker-fees)."""
        # Rond price af naar exchange tick-size, anders error 422
        rounded_price = self._round_price_to_tick(req.market, req.price, req.side)
        info = self.get_market_info(req.market)
        notional_dec = int(info.get('notionalDecimals', 2))
        qty_dec = int(info.get('quantityDecimals', 8))
        body = {
            'market': req.market,
            'side': req.side,
            'orderType': 'limit',
            'amount': f"{req.amount:.{qty_dec}f}",
            'price': f"{rounded_price:.{notional_dec}f}",
        }
        logger.info(f"  price ronded €{req.price:.2f} → €{rounded_price:.{notional_dec}f} "
                    f"(tick=€{float(info.get('tickSize', '0.01'))})")
        if req.post_only:
            body['postOnly'] = True
        if req.operator_id:
            body['operatorId'] = req.operator_id
        
        r = self._request('POST', '/order', body=body, auth=True)
        return OrderResponse(
            order_id=r.get('orderId', ''),
            status=r.get('status', ''),
            filled_amount=float(r.get('filledAmount', 0)),
            filled_amount_quote=float(r.get('filledAmountQuote', 0)),
            fee_paid=float(r.get('feePaid', 0)),
            fee_currency=r.get('feeCurrency', ''),
            raw=r,
        )
    
    def cancel_order(self, market: str, order_id: str) -> dict:
        return self._request('DELETE', '/order',
                              params={'market': market, 'orderId': order_id},
                              auth=True)
    
    def get_order(self, market: str, order_id: str) -> dict:
        return self._request('GET', '/order',
                              params={'market': market, 'orderId': order_id},
                              auth=True)
    
    def wait_for_fill_or_cancel(self, market: str, order_id: str,
                                  timeout_sec: int = 300,
                                  poll_interval_sec: int = 30) -> dict:
        """Poll order status. If not filled binnen timeout: cancel.
        
        Returns final order state dict met velden:
          - status: 'filled', 'partiallyFilled', 'canceled', 'rejected', etc
          - filledAmount: actual BTC filled
          - filledAmountQuote: actual EUR exchanged
          - feePaid, feeCurrency
        """
        deadline = time.time() + timeout_sec
        last_status = None
        while time.time() < deadline:
            order = self.get_order(market, order_id)
            status = order.get('status', '')
            filled = float(order.get('filledAmount', 0))
            amount = float(order.get('amount', 0))
            
            if status != last_status:
                logger.info(f"Order {order_id[:8]}... status={status} "
                            f"filled={filled:.8f}/{amount:.8f}")
                last_status = status
            
            # Terminal states
            if status in ('filled', 'canceled', 'rejected', 'expired'):
                return order
            # Partial fill is also "done enough" if we're at timeout-edge
            
            time.sleep(poll_interval_sec)
        
        # Timeout — cancel
        logger.warning(f"Order {order_id} timeout na {timeout_sec}s. Cancellen...")
        try:
            self.cancel_order(market, order_id)
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
        # Re-fetch finale status (kan zijn dat fill alsnog kwam tijdens cancel-request)
        return self.get_order(market, order_id)
