#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Users/sakesaakstra/Desktop/crypto_hartrace"
PYTHON="/opt/anaconda3/bin/python"

# Geen kill-switch hier — data update is read-only, mag altijd
cd "$PROJECT_ROOT"
exec "$PYTHON" -u scripts/btc_daily_data_update.py "$@"
