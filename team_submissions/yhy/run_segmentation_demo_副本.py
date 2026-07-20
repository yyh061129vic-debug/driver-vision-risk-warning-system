import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


DEFAULT_INPUT_DIR = Path("data_raw/sample_images")
DEFAULT_OUTPUT_DIR = Path("outputs/segmentation_demo")
DEFAULT_VIDEO_DIR = Path("data_raw/videos")

DEFAULT_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DEFAULT_RISK_MODE = "road-only"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mov",
    ".mp4",
}

ROAD_OVERLAY_ALPHA = 0.35
RISK_OVERLAY_ALPHA = 0.55

MIN_OBSTACLE_AREA = 500
MAX_OBSTACLE_AREA_RATIO = 0.65
BOTTOM_ROI_START = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple SegFormer road segmentation demo "
            "on images or video."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Video path. Recommended folder: "
            f"{DEFAULT_VIDEO_DIR}."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Image folder used when --video is omitted."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for result images, videos, and CSV reports.",
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face SegFormer model name.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional road confidence threshold. "
            "Use 0 to keep the raw class prediction."
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Process every Nth video frame. "
            "Use 1 for best visual continuity."
        ),
    )

    parser.add_argument(
        "--risk-mode",
        choices=[
            "road-only",
            "heuristic",
        ],
        default=DEFAULT_RISK_MODE,
        help=(
            "road-only shows road segmentation without "
            "heuristic obstacle warnings. heuristic keeps "
            "the old red suspicious-region overlay."
        ),
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def find_road_class_id(
    model: SegformerForSemanticSegmentation,
) -> int:
    id_to_label = model.config.id2label

    for class_id, label in id_to_label.items():
        normalized_label = str(label).lower().strip()

        if normalized_label == "road":
            print(
                "Road class found automatically: "
                f"id={class_id}, label={label}"
            )
            return int(class_id)

    for class_id, label in id_to_label.items():
        normalized_label = str(label).lower()

        if "road" in normalized_label:
            print(
                "Road-like class found automatically: "
                f"id={class_id}, label={label}"
            )
            return int(class_id)

    raise RuntimeError(
        "Road class was not found in the model label mapping."
    )


def load_model(
    model_name: str,
) -> tuple[
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    torch.device,
    int,
]:
    print(f"Loading SegFormer model: {model_name}")

    processor = SegformerImageProcessor.from_pretrained(
        model_name
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name
    )

    device = select_device()
    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    road_class_id = find_road_class_id(model)

    return (
        processor,
        model,
        device,
        road_class_id,
    )


def keep_largest_road_components(
    road_mask: np.ndarray,
    maximum_components: int = 3,
) -> np.ndarray:
    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            road_mask,
            connectivity=8,
        )
    )

    if component_count <= 1:
        return road_mask

    components = []

    for component_id in range(1, component_count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )
        components.append(
            (component_id, area)
        )

    components.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    filtered_mask = np.zeros_like(road_mask)

    for component_id, _ in components[:maximum_components]:
        filtered_mask[
            labels == component_id
        ] = 255

    return filtered_mask


def clean_road_mask(
    road_mask: np.ndarray,
) -> np.ndarray:
    road_mask = np.where(
        road_mask > 0,
        255,
        0,
    ).astype(np.uint8)

    road_mask = cv2.medianBlur(
        road_mask,
        5,
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (17, 17),
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return keep_largest_road_components(
        road_mask,
        maximum_components=3,
    )


def create_region_of_interest(
    height: int,
    width: int,
) -> np.ndarray:
    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    top_y = int(height * BOTTOM_ROI_START)

    polygon = np.array(
        [
            [
                (int(width * 0.18), top_y),
                (int(width * 0.82), top_y),
                (int(width * 0.98), height - 1),
                (int(width * 0.02), height - 1),
            ]
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        roi_mask,
        polygon,
        255,
    )

    return roi_mask


def create_drivable_envelope(
    road_mask: np.ndarray,
) -> np.ndarray:
    height, width = road_mask.shape

    contours, _ = cv2.findContours(
        road_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros_like(road_mask)

    useful_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_OBSTACLE_AREA
    ]

    if not useful_contours:
        return np.zeros_like(road_mask)

    all_points = np.vstack(useful_contours)
    hull = cv2.convexHull(all_points)

    envelope = np.zeros_like(road_mask)

    cv2.fillConvexPoly(
        envelope,
        hull,
        255,
    )

    horizontal_kernel_width = max(
        31,
        int(width * 0.07),
    )

    if horizontal_kernel_width % 2 == 0:
        horizontal_kernel_width += 1

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            horizontal_kernel_width,
            13,
        ),
    )

    envelope = cv2.dilate(
        envelope,
        horizontal_kernel,
        iterations=1,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    )

    envelope = cv2.morphologyEx(
        envelope,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    roi_mask = create_region_of_interest(
        height,
        width,
    )

    return cv2.bitwise_and(
        envelope,
        roi_mask,
    )


def detect_suspicious_regions(
    road_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, int, int, int]],
]:
    height, width = road_mask.shape

    drivable_envelope = create_drivable_envelope(
        road_mask
    )

    suspicious_mask = cv2.bitwise_and(
        drivable_envelope,
        cv2.bitwise_not(road_mask),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    suspicious_mask = cv2.morphologyEx(
        suspicious_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    suspicious_mask = cv2.morphologyEx(
        suspicious_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            suspicious_mask,
            connectivity=8,
        )
    )

    final_mask = np.zeros_like(suspicious_mask)
    valid_boxes = []

    minimum_area = max(
        MIN_OBSTACLE_AREA,
        int(height * width * 0.001),
    )
    maximum_area = int(
        height * width * MAX_OBSTACLE_AREA_RATIO
    )

    for component_id in range(1, component_count):
        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )
        box_width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        box_height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < minimum_area:
            continue

        if area > maximum_area:
            continue

        if box_width < 18 or box_height < 18:
            continue

        aspect_ratio = box_width / box_height

        if aspect_ratio > 10 and box_height < 35:
            continue

        final_mask[
            labels == component_id
        ] = 255

        valid_boxes.append(
            (
                x,
                y,
                box_width,
                box_height,
                area,
            )
        )

    return (
        final_mask,
        valid_boxes,
    )


def calculate_center_overlap(
    suspicious_mask: np.ndarray,
) -> float:
    height, width = suspicious_mask.shape

    center_mask = np.zeros_like(suspicious_mask)
    center_mask[
        int(height * 0.45):,
        int(width * 0.30):int(width * 0.70),
    ] = 255

    suspicious_pixels = int(
        np.count_nonzero(suspicious_mask)
    )

    if suspicious_pixels == 0:
        return 0.0

    center_pixels = int(
        np.count_nonzero(
            cv2.bitwise_and(
                suspicious_mask,
                center_mask,
            )
        )
    )

    return center_pixels / suspicious_pixels * 100


def calculate_bottom_overlap(
    suspicious_mask: np.ndarray,
) -> float:
    height, _ = suspicious_mask.shape

    bottom_mask = np.zeros_like(suspicious_mask)
    bottom_mask[
        int(height * 0.65):,
        :,
    ] = 255

    suspicious_pixels = int(
        np.count_nonzero(suspicious_mask)
    )

    if suspicious_pixels == 0:
        return 0.0

    bottom_pixels = int(
        np.count_nonzero(
            cv2.bitwise_and(
                suspicious_mask,
                bottom_mask,
            )
        )
    )

    return bottom_pixels / suspicious_pixels * 100


def determine_risk_level(
    road_ratio: float,
    obstacle_ratio: float,
    obstacle_count: int,
    largest_obstacle_ratio: float,
    center_overlap_ratio: float,
    bottom_overlap_ratio: float,
) -> str:
    if road_ratio < 5.0:
        return "UNKNOWN"

    if obstacle_count == 0:
        return "LOW"

    if (
        obstacle_ratio < 1.5
        and largest_obstacle_ratio < 0.8
    ):
        return "LOW"

    high_risk_conditions = [
        obstacle_ratio >= 8.0,
        largest_obstacle_ratio >= 4.0,
        center_overlap_ratio >= 20.0,
        bottom_overlap_ratio >= 15.0,
    ]

    if sum(high_risk_conditions) >= 3:
        return "HIGH"

    medium_risk_conditions = [
        obstacle_ratio >= 2.0,
        largest_obstacle_ratio >= 1.0,
        center_overlap_ratio >= 10.0,
        bottom_overlap_ratio >= 8.0,
    ]

    if sum(medium_risk_conditions) >= 2:
        return "MEDIUM"

    return "LOW"


def get_risk_text_color(
    risk_level: str,
) -> tuple[int, int, int]:
    if risk_level == "HIGH":
        return (0, 0, 255)

    if risk_level == "MEDIUM":
        return (0, 165, 255)

    if risk_level == "UNKNOWN":
        return (255, 255, 255)

    return (0, 255, 0)


def create_visualization(
    original_rgb: np.ndarray,
    road_mask: np.ndarray,
    suspicious_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int, int]],
    risk_level: str,
) -> np.ndarray:
    visualization = original_rgb.astype(np.float32).copy()

    road_pixels = road_mask > 0
    suspicious_pixels = suspicious_mask > 0

    road_color = np.zeros_like(original_rgb)
    road_color[:, :] = [
        0,
        255,
        0,
    ]

    visualization[road_pixels] = (
        visualization[road_pixels]
        * (1.0 - ROAD_OVERLAY_ALPHA)
        + road_color[road_pixels]
        * ROAD_OVERLAY_ALPHA
    )

    risk_color = np.zeros_like(original_rgb)
    risk_color[:, :] = [
        255,
        0,
        0,
    ]

    visualization[suspicious_pixels] = (
        visualization[suspicious_pixels]
        * (1.0 - RISK_OVERLAY_ALPHA)
        + risk_color[suspicious_pixels]
        * RISK_OVERLAY_ALPHA
    )

    visualization = np.clip(
        visualization,
        0,
        255,
    ).astype(np.uint8)

    visualization_bgr = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR,
    )

    for (
        x,
        y,
        box_width,
        box_height,
        area,
    ) in boxes:
        cv2.rectangle(
            visualization_bgr,
            (x, y),
            (
                x + box_width,
                y + box_height,
            ),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            visualization_bgr,
            f"Suspicious: {area}px",
            (
                x,
                max(y - 8, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        visualization_bgr,
        (10, 10),
        (260, 52),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        visualization_bgr,
        f"Risk level: {risk_level}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        get_risk_text_color(risk_level),
        2,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(
        visualization_bgr,
        cv2.COLOR_BGR2RGB,
    )


def infer_road_mask(
    image: Image.Image,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    road_class_id: int,
    confidence_threshold: float,
) -> np.ndarray:
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(
            image.height,
            image.width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    segmentation_map = (
        upsampled_logits
        .argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if confidence_threshold > 0.0:
        confidence_map = (
            torch.nn.functional.softmax(
                upsampled_logits,
                dim=1,
            )
            .max(dim=1)[0][0]
            .cpu()
            .numpy()
        )

        road_mask = np.where(
            (
                segmentation_map == road_class_id
            )
            & (
                confidence_map >= confidence_threshold
            ),
            255,
            0,
        ).astype(np.uint8)
    else:
        road_mask = np.where(
            segmentation_map == road_class_id,
            255,
            0,
        ).astype(np.uint8)

    return clean_road_mask(road_mask)


def analyze_frame(
    original_rgb: np.ndarray,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    road_class_id: int,
    confidence_threshold: float,
    risk_mode: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    image = Image.fromarray(original_rgb).convert("RGB")

    road_mask = infer_road_mask(
        image=image,
        processor=processor,
        model=model,
        device=device,
        road_class_id=road_class_id,
        confidence_threshold=confidence_threshold,
    )

    if risk_mode == "heuristic":
        suspicious_mask, boxes = detect_suspicious_regions(
            road_mask
        )
    else:
        suspicious_mask = np.zeros_like(road_mask)
        boxes = []

    road_pixels = int(np.count_nonzero(road_mask))
    suspicious_pixels = int(
        np.count_nonzero(suspicious_mask)
    )
    total_pixels = int(road_mask.size)

    road_ratio = road_pixels / total_pixels * 100

    if road_pixels > 0:
        obstacle_ratio = (
            suspicious_pixels / road_pixels * 100
        )
    else:
        obstacle_ratio = 0.0

    largest_obstacle_area = 0

    if boxes:
        largest_obstacle_area = max(
            box[4]
            for box in boxes
        )

    largest_obstacle_ratio = (
        largest_obstacle_area
        / total_pixels
        * 100
    )

    center_overlap_ratio = calculate_center_overlap(
        suspicious_mask
    )
    bottom_overlap_ratio = calculate_bottom_overlap(
        suspicious_mask
    )

    if risk_mode == "heuristic":
        risk_level = determine_risk_level(
            road_ratio=road_ratio,
            obstacle_ratio=obstacle_ratio,
            obstacle_count=len(boxes),
            largest_obstacle_ratio=largest_obstacle_ratio,
            center_overlap_ratio=center_overlap_ratio,
            bottom_overlap_ratio=bottom_overlap_ratio,
        )
    elif road_ratio < 5.0:
        risk_level = "UNKNOWN"
    else:
        risk_level = "LOW"

    result_rgb = create_visualization(
        original_rgb=original_rgb,
        road_mask=road_mask,
        suspicious_mask=suspicious_mask,
        boxes=boxes,
        risk_level=risk_level,
    )

    metrics = {
        "road_ratio_percent": f"{road_ratio:.2f}",
        "suspicious_ratio_percent": f"{obstacle_ratio:.2f}",
        "largest_region_percent": f"{largest_obstacle_ratio:.2f}",
        "center_overlap_percent": f"{center_overlap_ratio:.2f}",
        "bottom_overlap_percent": f"{bottom_overlap_ratio:.2f}",
        "suspicious_region_count": len(boxes),
        "risk_mode": risk_mode,
        "risk_level": risk_level,
    }

    return (
        metrics,
        result_rgb,
        road_mask,
        suspicious_mask,
    )


def save_report(
    rows: list[dict],
    report_path: Path,
) -> None:
    if not rows:
        return

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report saved to: {report_path}")


def process_image(
    image_path: Path,
    output_dir: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    road_class_id: int,
    confidence_threshold: float,
    risk_mode: str,
) -> dict:
    print(f"Processing image: {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    original_rgb = np.array(image)

    (
        metrics,
        result_rgb,
        road_mask,
        suspicious_mask,
    ) = analyze_frame(
        original_rgb=original_rgb,
        processor=processor,
        model=model,
        device=device,
        road_class_id=road_class_id,
        confidence_threshold=confidence_threshold,
        risk_mode=risk_mode,
    )

    result_path = output_dir / f"{image_path.stem}_result.jpg"
    road_mask_path = output_dir / f"{image_path.stem}_road_mask.png"
    risk_mask_path = output_dir / f"{image_path.stem}_risk_mask.png"

    Image.fromarray(result_rgb).save(
        result_path,
        quality=95,
    )
    Image.fromarray(road_mask).save(road_mask_path)
    Image.fromarray(suspicious_mask).save(risk_mask_path)

    print(f"Risk level: {metrics['risk_level']}")
    print(f"Saved result: {result_path}")

    return {
        "image": image_path.name,
        **metrics,
        "result_path": str(result_path),
    }


def process_images(
    input_dir: Path,
    output_dir: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    road_class_id: int,
    confidence_threshold: float,
    risk_mode: str,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}"
        )

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Found {len(image_paths)} image(s)")

    report_rows = []

    for image_path in image_paths:
        try:
            report_rows.append(
                process_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    processor=processor,
                    model=model,
                    device=device,
                    road_class_id=road_class_id,
                    confidence_threshold=confidence_threshold,
                    risk_mode=risk_mode,
                )
            )
        except Exception as error:
            print(
                f"Failed to process {image_path.name}: {error}"
            )

    save_report(
        report_rows,
        output_dir / "segmentation_report.csv",
    )

    print("All available images processed")


def get_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer: {output_path}"
        )

    return writer


def process_video(
    video_path: Path,
    output_dir: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    road_class_id: int,
    confidence_threshold: float,
    frame_stride: int,
    risk_mode: str,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension: {video_path.suffix}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_video_path = (
        output_dir
        / f"{video_path.stem}_segmentation_result.mp4"
    )
    report_path = (
        output_dir
        / f"{video_path.stem}_video_report.csv"
    )

    writer = get_video_writer(
        output_path=output_video_path,
        fps=fps / frame_stride,
        frame_size=(width, height),
    )

    print(f"Processing video: {video_path}")
    print(f"Output video: {output_video_path}")

    frame_index = 0
    written_frame_count = 0
    report_rows = []

    try:
        while True:
            success, frame_bgr = capture.read()

            if not success:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            original_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            metrics, result_rgb, _, _ = analyze_frame(
                original_rgb=original_rgb,
                processor=processor,
                model=model,
                device=device,
                road_class_id=road_class_id,
                confidence_threshold=confidence_threshold,
                risk_mode=risk_mode,
            )

            result_bgr = cv2.cvtColor(
                result_rgb,
                cv2.COLOR_RGB2BGR,
            )
            writer.write(result_bgr)

            report_rows.append(
                {
                    "video": video_path.name,
                    "frame_index": frame_index,
                    "timestamp_seconds": f"{frame_index / fps:.3f}",
                    **metrics,
                }
            )

            written_frame_count += 1

            if written_frame_count % 10 == 0:
                print(
                    f"Processed {written_frame_count} output frame(s)"
                )

            frame_index += 1

    finally:
        capture.release()
        writer.release()

    save_report(
        report_rows,
        report_path,
    )

    print("Video processed")
    print(f"Saved video: {output_video_path}")


def main() -> None:
    args = parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError(
            "--confidence-threshold must be between 0 and 1"
        )

    print("Driver Vision Risk Warning System")

    (
        processor,
        model,
        device,
        road_class_id,
    ) = load_model(args.model_name)

    if args.video is not None:
        process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            device=device,
            road_class_id=road_class_id,
            confidence_threshold=args.confidence_threshold,
            frame_stride=args.frame_stride,
            risk_mode=args.risk_mode,
        )
        return

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        device=device,
        road_class_id=road_class_id,
        confidence_threshold=args.confidence_threshold,
        risk_mode=args.risk_mode,
    )


if __name__ == "__main__":
    main()
