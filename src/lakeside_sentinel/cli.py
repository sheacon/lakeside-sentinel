import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lakeside Sentinel — Detection & Alert System",
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

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--veh",
        action="store_true",
        help="Run vehicle detection (VEH) mode.",
    )
    mode_group.add_argument(
        "--hsp",
        action="store_true",
        help="Run high-speed person (HSP) detection mode.",
    )
    return parser.parse_args()
