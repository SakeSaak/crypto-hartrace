#!/bin/bash
# Wrapper voor btc_live_trader.py — geschikt voor launchd
# Logt naar stdout/stderr, launchd kan deze opvangen
set -euo pipefail

PROJECT_ROOT="/Users/sakesaakstra/Desktop/crypto_hartrace"
PYTHON="/opt/anaconda3/bin/python"

# Kill switch check op shell-niveau (extra laag)
if [ -f "$HOME/STOP_TRADING" ]; then
    echo "[$(date)] STOP_TRADING file present, aborting." >&2
    exit 0  # not error — operator-requested stop
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -u scripts/btc_live_trader.py "$@"
