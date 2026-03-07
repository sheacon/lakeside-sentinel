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
        "--review",
        action="store_true",
        help="Launch the review web app for human-in-the-loop review.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed DEBUG-level logging output.",
    )

    detector_group = parser.add_argument_group("single detector options (require --veh or --hsp)")
    detector_group.add_argument(
        "--veh",
        action="store_true",
        help="Run vehicle detection (VEH) mode.",
    )
    detector_group.add_argument(
        "--hsp",
        action="store_true",
        help="Run high-speed person (HSP) detection mode.",
    )
    detector_group.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude Vision verification of detections (requires ANTHROPIC_API_KEY).",
    )
    detector_group.add_argument(
        "--claude-keep-rejected",
        action="store_true",
        help="Keep rejected detections in the HTML report (default: remove them).",
    )

    args = parser.parse_args()

    if args.veh and args.hsp:
        parser.error("--veh and --hsp are mutually exclusive")

    if (args.veh or args.hsp) and args.review:
        parser.error("--veh/--hsp cannot be used with --review")

    if (args.claude or args.claude_keep_rejected) and not (args.veh or args.hsp):
        flag_names = []
        if args.claude:
            flag_names.append("--claude")
        if args.claude_keep_rejected:
            flag_names.append("--claude-keep-rejected")
        parser.error(f"{', '.join(flag_names)} can only be used with --veh or --hsp")

    return args
