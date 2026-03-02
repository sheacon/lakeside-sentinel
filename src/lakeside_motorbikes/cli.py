import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lakeside Motorbikes — Detection & Alert System",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Analyze the most recent daylight period (sunrise-to-sunset).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Backfill a specific date (YYYY-MM-DD). Analyzes that day's sunrise-to-sunset.",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help="Backfill mode: save all MP4 clips to a timestamped folder for manual inspection.",
    )
    parser.add_argument(
        "--scooter",
        action="store_true",
        help="Experimental: detect fast-moving people (scooter riders) via motion tracking.",
    )
    return parser.parse_args()
