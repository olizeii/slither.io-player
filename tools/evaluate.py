"""Evaluate the training logs: best runs, learning curve, what works, dashboard.

    python tools/evaluate.py                  # summary in the console + tuning/report.png
    python tools/evaluate.py --top 20 --show  # 20 best runs, open the dashboard window
    python tools/evaluate.py --only-new       # ignore archived logs of older bot versions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slither_bot.analysis import load_runs, plot, summary   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tuning", help="folder with episodes*.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--only-new", action="store_true", help="skip episodes_old_*.csv")
    ap.add_argument("--out", help="dashboard PNG (default: <dir>/report.png)")
    ap.add_argument("--show", action="store_true", help="open the dashboard window")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.dir, include_old=not args.only_new)
    print(summary(runs, top=args.top))
    if runs and not args.no_plot:
        try:
            out = plot(runs, args.out or Path(args.dir) / "report.png", show=args.show)
            print(f"\nDashboard: {out}")
        except ImportError:
            print("\n(for the dashboard: pip install matplotlib)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
