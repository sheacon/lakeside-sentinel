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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (requires --veh or --hsp).",
    )

    debug_group = parser.add_argument_group("debug mode options (require --debug)")
    debug_group.add_argument(
        "--veh",
        action="store_true",
        help="Run vehicle detection (VEH) mode.",
    )
    debug_group.add_argument(
        "--hsp",
        action="store_true",
        help="Run high-speed person (HSP) detection mode.",
    )
    debug_group.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude Vision verification of detections (requires ANTHROPIC_API_KEY).",
    )
    debug_group.add_argument(
        "--claude-keep-rejected",
        action="store_true",
        help="Keep rejected detections in the HTML report (default: remove them).",
    )

    args = parser.parse_args()

    debug_flags = [args.veh, args.hsp, args.claude, args.claude_keep_rejected]
    if args.debug:
        if not args.veh and not args.hsp:
            parser.error("--debug requires --veh or --hsp")
        if args.veh and args.hsp:
            parser.error("--veh and --hsp are mutually exclusive")
    elif any(debug_flags):
        flag_names = ["--veh", "--hsp", "--claude", "--claude-keep-rejected"]
        used = [f for f, v in zip(flag_names, debug_flags) if v]
        parser.error(f"{', '.join(used)} can only be used with --debug")

    return args
