"""Command-line runner for road and lane-boundary detection."""

# 作者：林展毅

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from 林展毅_road_lane_detector import DetectionResult, DetectorConfig, RoadLaneDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 0, path.name.lower())


def collect_images(folder: Path) -> list[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=_numeric_key)


def resolve_input_kind(path: Path) -> Literal["directory", "video"]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_dir():
        return "directory"
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported input path: {path}")


def _open_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v", "XVID"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            print(f"Video codec: {codec}")
            return writer
        writer.release()
    raise RuntimeError(f"Cannot create video writer: {output_path}")


def _read_image(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode image: {path}")
    return frame


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Cannot encode mask: {path}")
    encoded.tofile(path)


def _write_masks(
    masks_dir: Path | None,
    stem: str,
    result: DetectionResult,
) -> None:
    if masks_dir is None:
        return
    _save_mask(masks_dir / "road" / f"{stem}.png", result.road_mask)
    _save_mask(masks_dir / "lane" / f"{stem}.png", result.lane_mask)


def _report_progress(
    completed: int,
    total: int | None,
    result: DetectionResult,
) -> None:
    if completed % 20 != 0 and completed != total:
        return
    suffix = f"/{total}" if total is not None else ""
    print(
        f"Processed {completed}{suffix} frames | "
        f"{result.diagnostics.total_ms:.1f} ms | "
        f"lanes={result.diagnostics.lane_count} "
        f"confidence={result.diagnostics.lane_confidence:.2f}"
        + (" | lane=unknown" if result.diagnostics.lane_rejected else "")
    )


def process_image_directory(
    detector: RoadLaneDetector,
    input_path: Path,
    output_path: Path,
    masks_dir: Path | None = None,
    independent_images: bool = False,
) -> int:
    paths = collect_images(input_path)
    if not paths:
        raise ValueError(f"No readable images found in: {input_path}")

    first = _read_image(paths[0])
    expected_shape = first.shape
    writer = _open_writer(
        output_path,
        10.0,
        (first.shape[1], first.shape[0]),
    )
    try:
        for index, path in enumerate(paths):
            if independent_images:
                detector.reset_temporal_state()
            frame = first if index == 0 else _read_image(path)
            if frame.shape != expected_shape:
                raise ValueError(
                    f"Frame size changed at index {index}: "
                    f"expected {expected_shape}, got {frame.shape}"
                )
            result = detector.process_frame(frame)
            writer.write(result.overlay)
            _write_masks(masks_dir, path.stem, result)
            _report_progress(index + 1, len(paths), result)
    finally:
        writer.release()
    return len(paths)


def process_video(
    detector: RoadLaneDetector,
    input_path: Path,
    output_path: Path,
    masks_dir: Path | None = None,
) -> int:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, (width, height))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size changed at index {count}: "
                    f"expected {(height, width)}, got {frame.shape[:2]}"
                )
            result = detector.process_frame(frame)
            writer.write(result.overlay)
            _write_masks(masks_dir, f"{count:06d}", result)
            count += 1
            _report_progress(count, None, result)
    finally:
        capture.release()
        writer.release()
    if count == 0:
        raise ValueError(f"Video contains no readable frames: {input_path}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect road areas and longitudinal lane boundaries."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--road-model-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "林展毅_road_segformer_b2_bdd_best.pt",
        help="Optional local SegFormer road checkpoint (BDD two-class model).",
    )
    parser.add_argument(
        "--road-class-id",
        type=int,
        choices=(0, 1),
        help="Class index representing road (defaults to 1 for local checkpoints, otherwise 0).",
    )
    parser.add_argument(
        "--road-probability-threshold",
        type=float,
        default=0.20,
        help="Road probability threshold for a local two-class checkpoint.",
    )
    parser.add_argument(
        "--lane-road-probability-threshold",
        type=float,
        default=0.05,
        help="Optional lower road threshold used only to support lane search.",
    )
    parser.add_argument(
        "--independent-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset temporal state before each image in a directory.",
    )
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--hood-top-ratio", type=float, default=0.68)
    parser.add_argument(
        "--fixed-hood-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply a calibrated fixed hood crop (disabled by default).",
    )
    parser.add_argument(
        "--lane-bottom-ratio",
        type=float,
        default=0.96,
        help="Lower image boundary for lane search; independent of hood masking.",
    )
    parser.add_argument(
        "--lane-model-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "林展毅_lane_segformer_b0_bdd_best.pt",
    )
    parser.add_argument("--inference-height", type=int, default=384)
    parser.add_argument("--inference-width", type=int, default=672)
    parser.add_argument(
        "--road-inference-stride",
        type=int,
        default=1,
        help="Run road inference on every frame (use >1 only for a stable video stream).",
    )
    parser.add_argument(
        "--lane-inference-stride",
        type=int,
        default=1,
        help="Run lane inference on every frame (use >1 only for a stable video stream).",
    )
    parser.add_argument(
        "--legacy-lane-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the original raw HSV+Canny+Hough path for VIL100 comparison only.",
    )
    parser.add_argument(
        "--enable-surface-markings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge broad horizontal surface markings into lane output (disabled by default).",
    )
    parser.add_argument(
        "--enable-low-light-enhancement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enhance dark frames before learned lane inference (disabled by default).",
    )
    parser.add_argument(
        "--observed-markings-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only marking pixels actually observed in candidates (post-v11 experiment).",
    )
    parser.add_argument(
        "--enable-night-single-line-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recover one reflected night boundary (post-v11 experiment, disabled by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kind = resolve_input_kind(args.input)
        lane_model_path = (
            str(args.lane_model_path)
            if args.lane_model_path.exists()
            else None
        )
        detector = RoadLaneDetector(
            DetectorConfig(
                device=args.device,
                road_model_path=(
                    str(args.road_model_path) if args.road_model_path else None
                ),
                road_class_id=args.road_class_id,
                road_probability_threshold=args.road_probability_threshold,
                lane_road_probability_threshold=args.lane_road_probability_threshold,
                hood_top_ratio=args.hood_top_ratio,
                fixed_hood_crop=args.fixed_hood_crop,
                lane_bottom_ratio=args.lane_bottom_ratio,
                inference_height=args.inference_height,
                inference_width=args.inference_width,
                road_inference_stride=args.road_inference_stride,
                lane_inference_stride=args.lane_inference_stride,
                legacy_lane_mode=args.legacy_lane_mode,
                enable_surface_markings=args.enable_surface_markings,
                enable_low_light_enhancement=args.enable_low_light_enhancement,
                enable_night_single_line_recovery=(
                    args.enable_night_single_line_recovery
                ),
                observed_markings_only=args.observed_markings_only,
                lane_model_path=lane_model_path,
            )
        )
        if kind == "directory":
            frame_count = process_image_directory(
                detector,
                args.input,
                args.output,
                args.masks_dir,
                independent_images=args.independent_images,
            )
        else:
            frame_count = process_video(detector, args.input, args.output, args.masks_dir)
        print(f"Processed {frame_count} frames: {args.output}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
