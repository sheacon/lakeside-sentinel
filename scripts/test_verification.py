"""Diagnostic script — test Claude Vision verification against a video clip."""

from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

import anthropic
import cv2
import numpy as np

from lakeside_sentinel.detection.claude_verifier import _PROMPT
from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.detection.veh_detector import VEHDetector
from lakeside_sentinel.utils.image import crop_to_bbox, crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def encode_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float = 0.2,
) -> str:
    """Crop a detection and return base64-encoded JPEG string."""
    cropped = crop_to_bbox(frame, bbox, padding=padding)
    _, jpg_bytes = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpg_bytes.tobytes()).decode()


def run_verification(
    client: anthropic.Anthropic,
    image_b64: str,
    model: str,
    temperature: float,
    prompt: str,
) -> tuple[str, str]:
    """Send a single verification request to Claude.

    Returns:
        (verdict, raw_answer) — verdict is "confirmed" or "rejected".
    """
    response = client.messages.create(
        model=model,
        max_tokens=16,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    raw = response.content[0].text.strip().lower()
    verdict = "confirmed" if raw.startswith("yes") else "rejected"
    return verdict, raw


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Test Claude Vision verification against a video clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/test_verification.py --clip output/video/clip.mp4\n"
            "  python scripts/test_verification.py --clip clip.mp4 --runs 5 --save-crops\n"
            "  python scripts/test_verification.py --clip clip.mp4 --temperature 1.0"
        ),
    )
    parser.add_argument("--clip", type=Path, required=True, help="Path to MP4 clip")
    parser.add_argument("--runs", type=int, default=3, help="Verification runs per detection")
    parser.add_argument("--temperature", type=float, default=0.0, help="Claude temperature")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Claude model")
    parser.add_argument("--yolo-model", type=str, default="yolo26s.pt", help="YOLO model weights")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    parser.add_argument("--confidence", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--crop-padding", type=float, default=0.2, help="Bbox crop padding")
    parser.add_argument("--roi-y-start", type=float, default=0.0)
    parser.add_argument("--roi-y-end", type=float, default=1.0)
    parser.add_argument("--roi-x-start", type=float, default=0.0)
    parser.add_argument("--roi-x-end", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default=None, help="Override verification prompt")
    parser.add_argument(
        "--save-crops", action="store_true", default=False, help="Save crop images to output/"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    clip_path: Path = args.clip
    if not clip_path.exists():
        logger.error("Clip not found: %s", clip_path)
        raise SystemExit(1)

    prompt = args.prompt or _PROMPT

    # Extract and crop frames
    logger.info("Extracting frames from %s at %d FPS", clip_path, args.fps)
    mp4_bytes = clip_path.read_bytes()
    raw_frames = extract_frames(mp4_bytes, fps_sample=args.fps)
    frames = crop_to_roi(
        raw_frames,
        y_start=args.roi_y_start,
        y_end=args.roi_y_end,
        x_start=args.roi_x_start,
        x_end=args.roi_x_end,
    )
    logger.info("Extracted %d frames", len(frames))

    # Run YOLO detection
    logger.info("Running YOLO detection (model=%s, conf=%.2f)", args.yolo_model, args.confidence)
    detector = VEHDetector(model_name=args.yolo_model, confidence_threshold=args.confidence)
    _, class_best = detector.detect_detailed(frames)

    if not class_best:
        logger.info("No YOLO detections found — nothing to verify")
        return

    # Filter to above-threshold detections
    detections: dict[str, Detection] = {
        name: det for name, det in class_best.items() if det.confidence >= args.confidence
    }

    if not detections:
        logger.info("All detections below confidence threshold (%.2f)", args.confidence)
        # Show sub-threshold for reference
        for name, det in class_best.items():
            logger.info("  Sub-threshold: %s %.0f%%", name, det.confidence * 100)
        return

    # Prepare crops
    crops: dict[str, str] = {}
    for class_name, det in detections.items():
        crops[class_name] = encode_crop(det.frame, det.bbox, args.crop_padding)

    # Save crops if requested
    if args.save_crops:
        output_dir = Path("output/verification") / clip_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        for class_name, det in detections.items():
            cropped = crop_to_bbox(det.frame, det.bbox, padding=args.crop_padding)
            out_path = output_dir / f"{class_name.lower()}_crop.jpg"
            cv2.imwrite(str(out_path), cropped)
            logger.info("Saved crop: %s", out_path)

    # Run Claude verification
    client = anthropic.Anthropic()
    logger.info(
        "Running %d verification runs per detection (model=%s, temp=%.1f)",
        args.runs,
        args.model,
        args.temperature,
    )

    # Results table
    results: dict[str, list[tuple[str, str]]] = {}

    for class_name, image_b64 in crops.items():
        det = detections[class_name]
        results[class_name] = []
        logger.info("--- %s (YOLO confidence: %.0f%%) ---", class_name, det.confidence * 100)

        for run_idx in range(1, args.runs + 1):
            verdict, raw = run_verification(client, image_b64, args.model, args.temperature, prompt)
            results[class_name].append((verdict, raw))
            logger.info("  Run %d: %s (raw: %r)", run_idx, verdict, raw)

    # Summary
    print("\n=== Verification Summary ===")
    print(f"Model: {args.model} | Temperature: {args.temperature} | Runs: {args.runs}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()

    header = f"{'Class':<15} | {'YOLO Conf':>9} | {'Confirmed':>9} | {'Rejected':>8} | {'Rate':>6}"
    print(header)
    print("-" * len(header))

    for class_name, run_results in results.items():
        det = detections[class_name]
        confirmed = sum(1 for v, _ in run_results if v == "confirmed")
        rejected = sum(1 for v, _ in run_results if v == "rejected")
        rate = confirmed / len(run_results) * 100
        line = (
            f"{class_name:<15} | {det.confidence:>8.0%} | "
            f"{confirmed:>9} | {rejected:>8} | {rate:>5.0f}%"
        )
        print(line)

    print()


if __name__ == "__main__":
    main()
