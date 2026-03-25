"""
Kalshi Weather Bot — Auto Runner
==================================
Runs the scanner on a schedule, automatically placing bets
on the highest-edge weather markets.

Usage:
    python autorun.py                    # Run once now
    python autorun.py --schedule         # Run every 6 hours (matches NOAA update cycle)
    python autorun.py --schedule --hour 7  # Run daily at 7am
"""

import time
import argparse
import schedule
from datetime import datetime
from scanner import WeatherScanner, BetQueue, cprint

def run_scan(capital: float, min_edge: float, dry_run: bool):
    """One full scan + execute cycle."""
    cprint(f"\n{'='*65}", "cyan")
    cprint(f"  AUTO-RUN  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
    cprint(f"{'='*65}\n", "cyan")

    class Args:
        city       = None
        category   = None
        top        = 9
        auto_queue = True
        execute    = not dry_run
        verbose    = True
        demo       = dry_run

    args = Args()
    args.min_edge  = min_edge
    args.capital   = capital

    scanner = WeatherScanner()
    scanner.run(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule",  action="store_true", help="Run on a schedule")
    parser.add_argument("--interval",  type=int, default=6, help="Hours between scans (default: 6)")
    parser.add_argument("--hour",      type=int, default=None, help="Run daily at this hour (0-23)")
    parser.add_argument("--capital",   type=float, default=200.0)
    parser.add_argument("--min-edge",  type=float, default=0.10)
    parser.add_argument("--dry-run",   action="store_true", help="Simulate only, no real orders")
    args = parser.parse_args()

    fn = lambda: run_scan(args.capital, args.min_edge, args.dry_run)

    if not args.schedule:
        fn()
        return

    if args.hour is not None:
        cprint(f"  Scheduled: daily at {args.hour:02d}:00", "green")
        schedule.every().day.at(f"{args.hour:02d}:00").do(fn)
    else:
        cprint(f"  Scheduled: every {args.interval} hours", "green")
        schedule.every(args.interval).hours.do(fn)

    fn()  # Run once immediately
    cprint("  Scheduler running. Ctrl+C to stop.\n", "yellow")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
