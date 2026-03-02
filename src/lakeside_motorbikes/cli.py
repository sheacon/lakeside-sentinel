import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lakeside Motorbikes — Detection & Alert System",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Analyze a specific date (YYYY-MM-DD). Defaults to the most recent daylight period.",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send an email report (HTML without embedded videos).",
    )
    parser.add_argument(
        "--hsp",
        action="store_true",
        help="Experimental: detect high-speed persons (HSP) via motion tracking.",
    )
    return parser.parse_args()
