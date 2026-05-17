# Paper → Live Runbook

Stap-voor-stap procedure om de BTC-EUR ATR-trend bot van paper naar live te brengen. **Lees alles voor je begint.**

## Fase 0 — Pre-flight (vóór paper-installatie)

Verifieer dat alles ready is:

```bash
cd ~/Desktop/crypto_hartrace

# 1. Bot zelf test draait door
/opt/anaconda3/bin/python scripts/btc_live_trader.py --paper --no-notify

# 2. Verwachte output regels:
#    "ATR-stop signal: action=enter, ..."
#    "PAPER fill: buy ..." OF "HOLD: ..."
#    "action=paper-filled" OF "action=hold"

# 3. State file gemaakt
cat state/trader_state.json
# moet bevatten: btc_balance, eur_balance, atr_in_position, atr_current_stop

# 4. W8W.env permissies correct
ls -la ~/W8W.env
# moet zijn: -rw-------  (alleen jij kunt lezen)
```

Als alles werkt → door naar Fase 1.

---

## Fase 1 — Paper-validatie installeren (NU)

Doel: 1-2 weken paper-trading, automatisch elke avond, om vertrouwen op te bouwen vóór live geld.

### Stap 1.1 — Installeer beide launchd jobs

```bash
cd ~/Desktop/crypto_hartrace
cp launchd/com.sakesaakstra.btc-data-update.plist ~/Library/LaunchAgents/
cp launchd/com.sakesaakstra.btc-trader.plist ~/Library/LaunchAgents/

launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-data-update.plist
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
```

### Stap 1.2 — Verifieer dat jobs geregistreerd zijn

```bash
launchctl list | grep btc
# Verwacht: twee regels met 'com.sakesaakstra.btc-data-update' en 'com.sakesaakstra.btc-trader'
# PID kan '-' zijn (job niet actief draaiend) of een nummer (loopt nu)
```

### Stap 1.3 — Dagelijkse monitoring (week 1-2)

Elke ochtend even checken of de avond ervoor goed liep:

```bash
# Laatste trader-run log
tail -20 logs/trader_$(date -v-1d +%Y%m%d).log

# Welke beslissing werd genomen?
tail -1 logs/trades_$(date -v-1d +%Y%m%d).jsonl | python3 -m json.tool

# State check
cat state/trader_state.json
```

### Wat is "succesvolle" paper-validatie?

Na 1-2 weken moet je zien:
- ✓ Geen Python tracebacks in `logs/trader_*.log`
- ✓ State file blijft consistent (atr_in_position, atr_current_stop verandert logisch)
- ✓ Bij elke run actie={enter, hold, exit, paper-filled, of action=hold met 'binnen deadband'}
- ✓ Geen "BLOCKED" reasons behalve verwachte (kill switch, etc.)
- ✓ Beslissingen kloppen met handmatig gecheckte BTC-prijs vs MA50

Als IETS abnormaals: `touch ~/STOP_TRADING` om alles te pauzeren, dan onderzoeken.

---

## Fase 2 — Pre-flight checklist VÓÓR live

Voordat je `LIVE_TRADING=true` zet, doorloop deze checklist:

- [ ] Minimaal **1 week** paper draait zonder issues (idealiter 2)
- [ ] Je hebt minstens 1 trend-omslag in paper meegemaakt EN dat ging goed (entry of exit getriggerd)
- [ ] Je begrijpt elke regel in `logs/trader_*.log` en kunt verklaren waarom de bot deed wat hij deed
- [ ] Je hebt een **Bitvavo API key** met:
  - Permissions: **alleen** "View" en "Trade"
  - **GEEN** "Withdraw" permission (kritisch — beschermt tegen sleutel-diefstal)
  - IP-whitelist op je thuis-IP (check via `curl ifconfig.me`)
- [ ] Je accepteert dat je in worst-case scenario je €110 inzet kunt verliezen
- [ ] Je hebt fiscaal advies ingewonnen (NL: algorithmic trading kan in box 1 vallen i.p.v. box 3)
- [ ] Je weet hoe `touch ~/STOP_TRADING` werkt (testen vóór live!)
- [ ] Je hebt een EUR-startbalance op Bitvavo (minimaal €110 of meer voor buffer)

Als één checkbox niet aan: NIET live gaan.

---

## Fase 3 — Live conversie procedure

### Stap 3.1 — Test de kill switch eerst (op paper)

```bash
# Activeer kill switch
touch ~/STOP_TRADING

# Forceer een paper-run
/opt/anaconda3/bin/python scripts/btc_live_trader.py --paper --no-notify 2>&1 | grep -E "BLOCKED|STOP"
# Moet zien: "BLOCKED: Kill switch active" of vergelijkbaar

# Schoon weer maken
rm ~/STOP_TRADING
```

### Stap 3.2 — Update W8W.env met live credentials

Open `~/W8W.env` in een editor (NIET delen, NIET committen):

```
API_KEY=<jouw Bitvavo API key>
API_SECRET=<jouw Bitvavo API secret>
BITVAVO_OPERATOR_ID=<optioneel, leeg laten als geen>

# DEZE TWEE BEPALEN ALLES:
LIVE_TRADING=true     # was 'false'
DRY_RUN=false         # was 'true'

# Strategie (default = atr_trend, hoeft niet expliciet)
STRATEGY=atr_trend

# Risk caps (optioneel; defaults oké voor €110)
MAX_POSITION_EUR=110
```

Permissies dichthouden:
```bash
chmod 600 ~/W8W.env
ls -la ~/W8W.env  # moet -rw------- zijn
```

### Stap 3.3 — Eerst een dry-run test (geen orders, wel echte balance)

```bash
# Forceer dry-run vanaf command-line; env LIVE_TRADING=true wordt overschreven
/opt/anaconda3/bin/python scripts/btc_live_trader.py --dry-run

# Output moet tonen:
#  - Je échte Bitvavo BTC/EUR balance (niet de paper €110)
#  - "DRY-RUN: zou ... plaatsen" — geen echte order
```

Als dit klopt → door naar 3.4.
Als de balance onverwacht is → STOP, onderzoek.

### Stap 3.4 — Switch launchd-plist naar --live

```bash
# Edit de geïnstalleerde plist (NIET die in project-dir)
sed -i '' 's|<string>--paper</string>|<string>--live</string>|' \
    ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist

# Verifieer de wijziging
grep -A 5 ProgramArguments ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist

# Herstart de job zodat hij de nieuwe plist leest
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
```

### Stap 3.5 — Eerste live run handmatig

```bash
# Run nu manueel zodat je realtime ziet wat er gebeurt
/opt/anaconda3/bin/python scripts/btc_live_trader.py --live

# Kijk goed: dit plaatst ECHTE orders.
# Verwachte log:
#  "=== Run start ... mode=live strategy=atr_trend ==="
#  "ATR-stop signal: action=..."
#  "Placing limit order: buy 0.00... BTC @ €..."  (of sell/hold)
#  "Order filled" of "Order partially filled" of "Order cancelled (timeout)"
```

Eerste trade is binnen 5 minuten ofwel gevuld ofwel gecancelled (post-only limit).

### Stap 3.6 — Verifieer in Bitvavo zelf

Open de Bitvavo app of website, ga naar:
- **Orders** → moet je trade-historie tonen
- **Balance** → moet matchen met wat de bot logt

Mismatch = STOP en onderzoek.

---

## Fase 4 — Live operationeel

Vanaf nu draait alles automatisch. Wat je periodiek moet doen:

### Dagelijks (5 min)
```bash
# Was de avond-run OK?
tail -20 logs/trader_$(date +%Y%m%d).log

# Welke beslissing?
tail -1 logs/trades_$(date +%Y%m%d).jsonl | python3 -m json.tool
```

### Wekelijks (15 min)
```bash
# Aantal trades deze week
grep -l "filled\|exit\|enter" logs/trades_$(date +%Y%m%d).jsonl logs/trades_$(date -v-7d +%Y%m%d).jsonl 2>/dev/null

# Totale fees betaald
python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('logs/trades_*.jsonl'))[-7:]:
    for line in open(f):
        d = json.loads(line)
        total += d.get('fee_paid', 0) or 0
print(f'Fees laatste 7 dagen: €{total:.2f}')
"

# Vergelijk je werkelijke balance met simulatie-verwachting
```

### Per maand (30 min)
- Volledige backtest opnieuw draaien (`scripts/H2_TA_strategies.py`) om te checken dat ATR-stop nog steeds wint
- Performance live vs backtest-verwachting vergelijken (begin van post-thesis paper-data)

---

## Noodprocedures

### Stop alle handel onmiddellijk
```bash
touch ~/STOP_TRADING
# Bot zal volgende run direct stoppen vóór elke andere check
```

### Stop launchd-jobs helemaal
```bash
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-data-update.plist
```

### Terug naar paper (rollback)
```bash
# Edit W8W.env: LIVE_TRADING=false, DRY_RUN=true
# Switch plist:
sed -i '' 's|<string>--live</string>|<string>--paper</string>|' \
    ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist
```

### Cancel openstaande orders handmatig
Geen tooling daarvoor in deze bot. Doe het in Bitvavo Pro Mode UI direct (Orders → Cancel All).

---

## Belangrijke disclaimers

- **Geen financieel advies.** Dit is academische research-code voor je thesis.
- **Past performance ≠ future results.** Onze H2/H3 backtest geeft Sharpe 1.14 op 2022-2026 OOS data, maar dat is **geen garantie** voor de toekomst.
- **Crypto regimes shifteren.** ATR-stop werkt goed in trend-rijke periodes; in een choppy sideways regime kan het whipsaws geven.
- **De €110 schaal is bewust klein.** Verlies van het volledige bedrag is een acceptabel leertraject; verlies van een grotere positie is fundamenteel een ander gesprek.
- **Belasting NL:** algoritmische handel kan in box 1 vallen i.p.v. box 3. Praat met een fiscalist.

---

## Snelle referentie: commando's

| Wat | Commando |
|---|---|
| Pause direct | `touch ~/STOP_TRADING` |
| Hervat | `rm ~/STOP_TRADING` |
| Handmatige paper-run | `python scripts/btc_live_trader.py --paper` |
| Handmatige dry-run (echte balance, geen orders) | `python scripts/btc_live_trader.py --dry-run` |
| Handmatige live-run | `python scripts/btc_live_trader.py --live` |
| Logs vandaag | `tail -50 logs/trader_$(date +%Y%m%d).log` |
| State | `cat state/trader_state.json` |
| Disable schedule | `launchctl unload ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist` |
| Re-enable schedule | `launchctl load ~/Library/LaunchAgents/com.sakesaakstra.btc-trader.plist` |
