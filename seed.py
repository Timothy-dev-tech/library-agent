"""Create (or re-create) the simulated library database.

Usage:
    python seed.py                       # history ends today
    python seed.py --as-of 2026-11-16    # history ends on a chosen date
"""
import argparse
from datetime import date

from library_agent.config import DB_PATH
from library_agent.simulation import seed_database

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the simulated library database")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="last day of the simulated history (YYYY-MM-DD); default is today")
    args = parser.parse_args()
    summary = seed_database(DB_PATH, as_of=args.as_of)
    print(f"Database created at {DB_PATH}")
    for key, value in summary.items():
        print(f"  {key:14} {value}")