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
    return parser.parse_args()
