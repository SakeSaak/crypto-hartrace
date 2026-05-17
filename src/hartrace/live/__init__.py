"""Live trading module — BTC-EUR vol-managed on Bitvavo.

Architectuur:
  - config.py:     env-loader + validatie (DRY_RUN default, three-layer safety)
  - bitvavo_client.py:  Bitvavo REST API wrapper met HMAC-SHA256 auth
  - safety.py:     kill switch, position cap, sanity checks
  - forecaster.py: dagelijkse HAR-RS-Q-WE-X forecast
  - executor.py:   strategie-logica, order-besluit, execution

CRITICAL: lees scripts/live/README.md voor operator handleiding.
"""
__version__ = "0.1.0-skeleton"
