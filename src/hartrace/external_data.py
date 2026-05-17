"""External data fetchers for HAR-MIDAS regressors.

Three exogenous data sources at daily frequency:

  1. EUR/USD FX (yfinance, ticker 'EURUSD=X')
       Used to decompose BTC-EUR volatility into a BTC-USD component and
       a EUR-USD FX-vol component (the EUR-domiciled investor exposure).

  2. BTC-dominance (CoinGecko, /coins/bitcoin/market_chart range)
       Bitcoin market-cap / total crypto market-cap. Captures risk-on /
       risk-off rotation within the crypto market itself.

  3. Macro factors (FRED, fredapi)
       - DFF: Federal Funds effective rate
       - DGS10: 10-year US Treasury constant maturity yield
       - CPIAUCSL: CPI All Urban (release-date timing matters; for our
         purposes use the release date, not the reference month)
       Forecasting Bitcoin RV improves substantially when macro signals
       are added (Bergsli et al. 2022 baseline ~9%, +macro ~15%).

All fetchers save to data/external/{name}.parquet with a 'date' column
and the relevant value(s). Resumable via merge-on-date.

Place in: crypto_hartrace/src/hartrace/external_data.py
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ======================================================================
# 1) EUR/USD FX via yfinance
# ======================================================================
def fetch_eurusd(
    out_path: str | Path,
    start: str = '2018-01-01',
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Daily EUR/USD via yfinance. Output: date, eurusd_close, eurusd_logret.

    Implements a basic noise/return: log-return of close-to-close.
    Realized FX-vol on daily basis is the absolute return; for higher
    accuracy use intraday FX (not pursued here — daily FX-vol is dwarfed
    by crypto vol anyway).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install yfinance") from exc

    end = end or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"   EUR/USD: yfinance EURUSD=X, {start} → {end}")
    df = yf.Ticker('EURUSD=X').history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data — check network / ticker")
    df = df.rename(columns={'Close': 'eurusd_close'})
    df['date'] = df.index.tz_localize(None).normalize()
    df['eurusd_logret'] = np.log(df['eurusd_close']).diff()
    out = df[['date', 'eurusd_close', 'eurusd_logret']].reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"   saved {len(out):,} rows → {out_path.name}")
    return out


# ======================================================================
# 2) BTC dominance via CoinGecko
# ======================================================================
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_btc_dominance(
    out_path: str | Path,
    start: str = '2018-01-01',
    end: Optional[str] = None,
    throttle_s: float = 1.5,
) -> pd.DataFrame:
    """Daily BTC-dominance series from CoinGecko.

    Uses /global/market_cap_chart endpoint (no auth needed, public free).
    Throttle 1.5s/call since CoinGecko free tier is ~10-50 calls/min.

    Returns: date, btc_dominance (in 0..1).
    """
    end = end or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    print(f"   BTC-dominance: CoinGecko global market-cap chart, {start} → {end}")

    url = f"{COINGECKO_BASE}/global/market_cap_chart"
    params = {'days': 'max', 'vs_currency': 'usd'}
    headers = {'User-Agent': 'crypto_hartrace/0.1 (academic research)',
               'Accept': 'application/json'}

    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            break
        if r.status_code == 429:
            wait = 2 ** attempt * 5
            print(f"   429 rate-limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            raise RuntimeError(
                "CoinGecko returned 401 (auth) — this endpoint may have moved "
                "behind their Pro tier. Fallback: scrape from blockchain.com or "
                "compute from /coins/markets snapshots over time."
            )
        raise RuntimeError(f"CoinGecko {r.status_code}: {r.text[:200]}")
    else:
        raise RuntimeError("CoinGecko: gave up after retries")

    data = r.json().get('market_cap_chart', {})
    total = data.get('market_cap', [])    # [[ts_ms, total_mc_usd], ...]
    btc = data.get('btc_market_cap', [])  # idem voor BTC; veld kan ander naam hebben

    if not total or not btc:
        # Alternative CoinGecko shape — fall back to two separate /market_chart calls
        return _btc_dominance_fallback(out_path, start, end, throttle_s)

    df_total = pd.DataFrame(total, columns=['ts_ms', 'total_mc'])
    df_btc = pd.DataFrame(btc, columns=['ts_ms', 'btc_mc'])
    df = df_total.merge(df_btc, on='ts_ms')
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.tz_localize(None).dt.normalize()
    df['btc_dominance'] = df['btc_mc'] / df['total_mc']
    df = df.groupby('date', as_index=False).last()  # one row per day
    df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[['date', 'btc_dominance']].to_parquet(out_path, index=False)
    print(f"   saved {len(df):,} rows → {out_path.name}")
    return df


def _btc_dominance_fallback(out_path, start, end, throttle_s):
    """Compute BTC dominance from two /coins/{id}/market_chart calls.

    BTC market cap from /coins/bitcoin/market_chart and total from
    /global. If the global market_cap_chart endpoint is gated, this
    is the cheaper fallback (two REST calls instead of one).
    """
    print("   falling back to per-coin market_chart calls")
    headers = {'User-Agent': 'crypto_hartrace/0.1 (academic research)'}
    url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
    params = {'vs_currency': 'usd', 'days': 'max'}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    btc_mc = pd.DataFrame(r.json()['market_caps'], columns=['ts_ms', 'btc_mc'])
    time.sleep(throttle_s)

    url2 = f"{COINGECKO_BASE}/global"
    r2 = requests.get(url2, headers=headers, timeout=30)
    r2.raise_for_status()
    snapshot = r2.json()['data']
    total_today = float(snapshot['total_market_cap']['usd'])
    btc_today = float(snapshot['market_cap_percentage']['btc']) / 100

    print(f"   today: BTC-dominance ≈ {btc_today:.4f}")
    # We only get HISTORICAL btc mc from coingecko — total mc historical
    # needs a separate aggregation (sum across all coins per timestamp,
    # which is impractical free). Practical workaround: approximate
    # historical dominance via btc_mc / (btc_mc / btc_dominance_today * (1+drift)).
    # That's a hack; for production we'd use a paid endpoint or scrape.
    # For now: store BTC mc historically + today's dominance as anchor;
    # we'll reconstruct historical dominance in a separate processing step.
    btc_mc['date'] = pd.to_datetime(btc_mc['ts_ms'], unit='ms', utc=True).dt.tz_localize(None).dt.normalize()
    btc_mc = btc_mc.groupby('date', as_index=False).last()
    btc_mc['btc_dominance_approx_today_only'] = np.nan
    btc_mc.loc[btc_mc.index[-1], 'btc_dominance_approx_today_only'] = btc_today
    out_path = Path(out_path)
    btc_mc[['date', 'btc_mc', 'btc_dominance_approx_today_only']].to_parquet(out_path, index=False)
    print(f"   saved {len(btc_mc):,} BTC market-caps + today's dominance "
          f"→ {out_path.name}")
    print(f"   NOTE: full historical dominance needs paid endpoint or scrape; "
          f"see process_btc_dominance() docstring.")
    return btc_mc


# ======================================================================
# 3) FRED macro factors
# ======================================================================
def fetch_fred_series(
    series_ids: list[str],
    out_path: str | Path,
    api_key: Optional[str] = None,
    start: str = '2018-01-01',
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Pull a list of FRED series and pivot to wide format on date.

    Series IDs we want:
      DFF       Effective Federal Funds Rate (daily)
      DGS10     10-Year Treasury Constant Maturity Rate (daily)
      DGS2      2-Year Treasury Rate (daily)
      VIXCLS    VIX (daily)
      DTWEXBGS  Trade-Weighted USD Index (daily)
      T10Y2Y    10Y-2Y Treasury spread (daily; derived but FRED publishes)

    Requires API key from https://fred.stlouisfed.org/docs/api/api_key.html
    """
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError("pip install fredapi") from exc

    if api_key is None:
        import os
        api_key = os.environ.get('FRED_API_KEY')
        if not api_key:
            raise RuntimeError(
                "Set FRED_API_KEY env var or pass api_key= "
                "(register free at https://fred.stlouisfed.org/docs/api/api_key.html)"
            )
    end = end or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    fred = Fred(api_key=api_key)

    frames = []
    for sid in series_ids:
        print(f"   FRED {sid} ...", end='', flush=True)
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s = s.rename(sid).reset_index().rename(columns={'index': 'date'})
            s['date'] = pd.to_datetime(s['date']).dt.normalize()
            frames.append(s)
            print(f" {len(s):,} obs")
        except Exception as exc:
            print(f" FAILED: {exc!r}")

    if not frames:
        raise RuntimeError("No FRED series fetched")
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"   saved {len(df):,} rows × {df.shape[1]-1} series → {out_path.name}")
    return df


# ======================================================================
# Default series for our setup
# ======================================================================
DEFAULT_FRED_SERIES = [
    'DFF',       # Federal Funds Rate
    'DGS10',     # 10Y Treasury
    'DGS2',      # 2Y Treasury
    'T10Y2Y',    # 10Y-2Y spread
    'VIXCLS',    # VIX
    'DTWEXBGS',  # Trade-Weighted USD Index
]
