"""BTC-EUR vol-managed live trader — entry point.

Modes (alle default veilig):
  python btc_live_trader.py --paper      # simulated, geen API-keys nodig
  python btc_live_trader.py              # default = dry-run, leest balance, plaatst geen orders
  python btc_live_trader.py --dry-run    # expliciet dry-run
  python btc_live_trader.py --live       # ECHTE orders (vereist LIVE_TRADING=true + DRY_RUN=false in env)

Safety:
  Kill switch:  touch ~/STOP_TRADING
  Resume:       rm ~/STOP_TRADING
  
Schedule via launchd: zie launchd/com.sakesaakstra.btc-trader.plist
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hartrace.live.config import load_config
from hartrace.live.executor import Executor


def setup_logging(log_dir: Path, verbose: bool = False):
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = log_dir / f'trader_{date_str}.log'
    
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logging.info(f"Logging to {log_file}")


def macos_notify(title: str, message: str):
    """Send macOS notification (best effort)."""
    import subprocess
    try:
        subprocess.run([
            'osascript', '-e',
            f'display notification "{message}" with title "{title}"'
        ], capture_output=True, timeout=5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='BTC-EUR vol-managed trader')
    parser.add_argument('--paper', action='store_true',
                         help='Paper trading (geen API-keys, gesimuleerde fills)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                         help='Dry-run mode (leest balance, plaatst geen orders)')
    parser.add_argument('--live', action='store_true', default=False,
                         help='Live trading (vereist LIVE_TRADING=true in env)')
    parser.add_argument('--env', default=str(Path.home() / 'W8W.env'),
                         help='Path naar env-bestand met API credentials')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-notify', action='store_true',
                         help='Suppress macOS notifications')
    args = parser.parse_args()
    
    if args.paper and args.live:
        print("ERROR: --paper en --live kunnen niet samen", file=sys.stderr)
        sys.exit(2)
    
    project_root = Path(__file__).parent.parent
    log_dir = project_root / 'logs'
    setup_logging(log_dir, args.verbose)
    
    # Load config (skip key requirement if paper mode)
    cfg = load_config(
        env_path=Path(args.env),
        require_keys=not args.paper,
    )
    
    # CRITICAL: only set cli_live=True if --live was explicit
    cli_live = args.live and not args.dry_run and not args.paper
    
    executor = Executor(cfg, cli_live=cli_live, paper=args.paper)
    
    try:
        decision = executor.run_once()
        
        # Notification
        if not args.no_notify:
            summary = f"σ̂={decision.sigma_pred:.4f}, w*={decision.target_weight:.2f}, "
            if decision.action in ('buy', 'sell', 'paper-filled'):
                summary += f"{decision.action} €{abs(decision.delta_eur):.2f}"
            elif decision.action == 'blocked':
                summary += f"BLOCKED: {decision.block_reason[:40]}"
            else:
                summary += decision.action
            macos_notify(
                f"BTC Trader [{decision.mode}]",
                summary,
            )
        
        logging.info(f"=== Run complete: action={decision.action} ===")
        sys.exit(0)
    
    except Exception as e:
        logging.exception(f"FATAL: {e}")
        if not args.no_notify:
            macos_notify("BTC Trader ERROR", str(e)[:100])
        sys.exit(1)


if __name__ == '__main__':
    main()
