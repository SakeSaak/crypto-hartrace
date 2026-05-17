# Changelog

## [2026-05-17] Strategy switch: ATR-stop → Pure Trend MA50

Empirische basis: `scripts/H8_robustness_extensions.py` (zie analyse 1-4)

### Wat is veranderd?
- **Default strategy** in `src/hartrace/live/config.py`:
  - Was: `atr_trend` (ATR-stop + MA50 trailing stop)
  - Nu: `pure_trend` (pure MA50 crossover, geen trailing stop)
- Beide locaties (dataclass default + load_config fallback) bijgewerkt
- ATR-strategy code blijft beschikbaar via `STRATEGY=atr_trend` env override

### Waarom?
Vier convergente bevindingen uit H8-robustheids-extensies (1 312-dag OOS sample):

| Test | Pure trend MA50 | ATR-stop + MA50 |
|---|---|---|
| Full sample Sharpe | +0.92 | +0.68 |
| Stress 2024-2026 Sharpe | +0.36 | +0.01 |
| MA-window robustness | MA50 optimaal (peak inverted-U) | MA50 ook beste, vlakker plateau |
| Fee-drag | 25.4% | 28.7% |

Pure trend was empirisch beter op alle vier de meetlatten, met name in het stress-regime waar de ATR-strategie naar break-even degradeerde.

### Wat gebeurt er met de huidige positie?
- Huidige state (17 mei 2026): 0.00140610 BTC, ATR-stop=€62 483
- Onder pure_trend: state-velden `atr_in_position` en `atr_current_stop` worden genegeerd
- Bij volgende cron-run (22:05 UTC):
  - Als BTC > MA50 (huidig: €67 577 > €64 353 ✓): positie behouden, target_w=0.90
  - Als BTC < MA50: positie verkocht, target_w=0
- **Geen abrupte liquidatie** bij de switch zelf

### Rollback procedure
Als pure_trend in live trading slechter blijkt dan verwacht:
```bash
cd /Users/sakesaakstra/Desktop/crypto_hartrace
git revert HEAD       # of git checkout HEAD~1 src/hartrace/live/config.py
```
ATR-state in `state/trader_state.json` is bewaard gebleven; rollback herstelt de strategie zonder data-verlies.

### Eerste live run met nieuwe strategie
Verwacht: 18 mei 2026, 22:05 UTC, via launchd `com.sakesaakstra.btc-trader.plist`
