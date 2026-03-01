import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lakeside Motorbikes — Detection & Alert System",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Download and analyze all events from the past 24 hours instead of live monitoring.",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help="Backfill mode: save all MP4 clips to a timestamped folder for manual inspection.",
    )
    return parser.parse_args()
