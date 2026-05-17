# BTC Live Trader — Operator Handleiding

**Strategy: pure_trend (MA50)** — empirische winner uit G5 robustness-checks.

## Strategiebeschrijving

Op basis van G3-G5 analyses gekozen:
- **Regel**: positie = 90% in BTC als prijs > 50-dagen moving average, anders 0% (cash).
- **Rebalance trigger**: alleen wanneer |target_w − current_w| > 5% (deadband, voorkomt churn).
- **Verwacht**: Sharpe ≈ 1.03 net van fees, AnnRet ~30%, MDD -29%, ~21 rebalances/jaar.
- **Robustness**: validated over twee sub-periodes (early bull 2022-2024, late bear 2024-2026) en cross-asset op ETH-EUR.

Het HAR-RS-DOW vol-model uit de thesis is **niet** onderdeel van de live trading-logic, omdat G4-G5 toonden dat simpele price-trend MA50 robuust de hybrid en pure vol-managed strategieën verslaat. HAR blijft beschikbaar voor analyse — switch met `STRATEGY=vol_managed` in env als gewenst.

## Veiligheidsfilosofie

Drie onafhankelijke veiligheidslagen voor live trading. ALLE drie moeten op groen staan:

1. **`LIVE_TRADING=true`** in `~/W8W.env` (env-laag)
2. **`DRY_RUN=false`** in `~/W8W.env` (env-laag, default-safe)
3. **`--live`** command-line vlag bij elke run (runtime-laag)

Plus operationele kill-switches:

- `touch ~/STOP_TRADING` — pauzeert direct (shell-laag én Python-laag)
- `MAX_POSITION_EUR=1000` — hard code cap op totale BTC-exposure (env-overridable)
- `MAX_DAILY_TURNOVER_PCT=1.0` voor pure_trend (default; trend-flips zijn legitieme grote bewegingen)

## W8W.env volledig overzicht

```
# Bitvavo credentials (vereist voor live/dry-run, optioneel voor paper)
API_KEY=jouw_bitvavo_api_key
API_SECRET=jouw_bitvavo_api_secret
BITVAVO_OPERATOR_ID=optioneel

# Live trading flags (default-safe)
LIVE_TRADING=false       # 'true' nodig voor echte orders
DRY_RUN=true             # 'false' nodig voor echte orders

# Strategy (optioneel; default = pure_trend)
STRATEGY=pure_trend      # of 'vol_managed' voor HAR-vol-managed strategie

# Risk caps (optioneel; defaults toegepast als afwezig)
MAX_POSITION_EUR=110     # absolute cap op BTC-exposure
MAX_DAILY_TURNOVER_PCT=1.0  # voor pure_trend: 1.0 toestaan voor trend-flips
```

Daarna lockdown: `chmod 600 ~/W8W.env`

## Initiële setup

### Stap 1 — Bitvavo API key

Op Bitvavo:
1. Account → API → "Nieuwe API key"
2. Permissions: **alleen** "View" en "Trade" — **NIET** "Withdraw"
3. IP whitelist: jouw thuis-IP (verifieer via `curl ifconfig.me`)
4. Bewaar de secret — wordt eenmalig getoond

### Stap 2 — Test paper mode (geen keys nodig)

```bash
cd ~/Desktop/crypto_hartrace
/opt/anaconda3/bin/python scripts/btc_live_trader.py --paper
```

Run dit een paar dagen handmatig. Output toont:
- Trend signal (price, MA50, deviation, in_trend bool)
- Target weight (0.90 of 0.00)
- Portfolio state, delta
- Paper-fill of HOLD

State staat in `state/trader_state.json`, logs in `logs/trades_*.jsonl`.

### Stap 3 — Schedule via launchd (paper mode)

```bash
# Update plist om --paper te gebruiken
sed -i '' 's|<string>--dry-run</string>|<string>--paper</string>|' \
    launchd/com.sakesaakstra.btc-trader.plist

# Installeer beide jobs (data update + trader)
cp launchd/com.sakesaakstra.btc-data-update.plist ~/Library/LaunchAgents/
cp launchd/com.sakesaakstra.btc-trader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-data-update.plist
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
```

Vanaf nu: dagelijks om 22:00 UTC verse candles ophalen, 22:05 UTC paper-trader-besluit.
**Laat dit minimaal 1-2 weken paper draaien** om vertrouwen op te bouwen.

### Stap 4 — Switch naar live (NA paper-validatie)

```bash
# 1. Activeer live in env
sed -i '' 's/^LIVE_TRADING=.*/LIVE_TRADING=true/' ~/W8W.env
sed -i '' 's/^DRY_RUN=.*/DRY_RUN=false/' ~/W8W.env

# 2. Switch launchd vlag
sed -i '' 's|<string>--paper</string>|<string>--live</string>|' \
    ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist

# 3. Reload
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
```

**Voor de eerste trade**: zorg dat je BTC-positie ongeveer overeenkomt met wat de bot zou willen. Bij `in_trend=True` wil hij ~90% BTC. Als je nu 100% EUR hebt en de trend is on, de bot zal in één klap willen kopen. Dat is OK met `MAX_DAILY_TURNOVER_PCT=1.0` voor pure_trend.

## Operationele commando's

```bash
# Kill switch
touch ~/STOP_TRADING       # stoppen
rm ~/STOP_TRADING          # hervatten

# Disable schedule
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist

# Re-enable
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist

# Logs van vandaag
tail -50 logs/trader_$(date +%Y%m%d).log

# Handmatige run (een keer)
/opt/anaconda3/bin/python scripts/btc_live_trader.py --paper

# Forceer dry-run met echte balance (geen orders)
/opt/anaconda3/bin/python scripts/btc_live_trader.py --dry-run
```

## Wat als er iets misgaat

**De bot wil een onverwachte trade plaatsen**: `touch ~/STOP_TRADING` direct. Cancel openstaande orders handmatig in Bitvavo Pro Mode. Check `logs/trades_*.jsonl` om de besluitvorming te reconstrueren.

**Trend MA50 lijkt achterhaald**: de moving-average heeft inherent lag. In flash crashes is de eerste paar dagen het signaal "still in trend" terwijl prijs al hard zakt. Dit is bewust — anders krijg je whipsaw-handel. Accepteer 5-10% drawdown na een trend-top.

**Data-update faalt**: check `logs/data_update_*.log`. Meestal is dit een Bitvavo rate-limit (wacht 1 uur) of API-outage (gewoon morgen retry).

**MA50-signaal flipt te vaak**: dit zou abnormaal zijn. Onze backtest gaf ~21 trades/jaar. Als je meer dan 5/maand ziet, zit er iets mis met de prijsdata.

## Disclaimers

- **Geen advies**: academische research-code, geen financieel advies.
- **Past performance ≠ future**: backtest-Sharpe is geen garantie. Onze G5 toont dat trend MA50 robuust is in OOS-sub-periodes, maar dit is geen guarantee voor toekomstige regimes.
- **Belasting NL**: pure trend rebalancing kan in box-1 vallen i.p.v. box-3. Praat met een fiscalist voor je live gaat.
- **MiCA / DNB**: Bitvavo is gereguleerd. Persoonlijk handelen via API is toegestaan; commerciële algos vereisen extra registratie.
- **Sleutelbeheer**: als iemand je API-keys steelt, kunnen ze met "Trade" permission je BTC verkopen voor EUR (maar niet withdrawen als je dat permission niet gaf). Bewaar `W8W.env` met `chmod 600` en backup niet naar cloud.
