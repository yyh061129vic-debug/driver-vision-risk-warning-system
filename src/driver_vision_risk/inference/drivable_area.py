"""任务 5 可行驶区域 Demo 的图像与视频推理入口。"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from driver_vision_risk.models.segformer import (
    DrivableAreaPrediction,
    SegformerDrivableAreaSegmenter,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次性把大文件读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_overlay(
    image: Image.Image,
    prediction: DrivableAreaPrediction,
    road_color: tuple[int, int, int],
    boundary_color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    """把道路区域半透明着色，并用实色标出道路内边界。"""

    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    color = np.asarray(road_color, dtype=np.float32)
    # 只修改道路像素，非道路区域保持原始图像不变。
    base[prediction.mask] = base[prediction.mask] * (1.0 - alpha) + color * alpha
    base[prediction.boundary] = np.asarray(boundary_color, dtype=np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _prediction_metrics(prediction: DrivableAreaPrediction) -> dict[str, float | int | str]:
    """汇总单帧延迟、道路覆盖率以及道路区域内的平均置信度。"""

    road_pixels = int(prediction.mask.sum())
    pixel_count = int(prediction.mask.size)
    mean_confidence = float(prediction.confidence[prediction.mask].mean()) if road_pixels else 0.0
    return {
        "device": prediction.device,
        "latency_ms": round(prediction.latency_ms, 3),
        "pixel_count": pixel_count,
        "road_pixel_count": road_pixels,
        "road_coverage": round(road_pixels / pixel_count, 6),
        "mean_road_confidence": round(mean_confidence, 6),
    }


def _base_metadata(
    input_path: Path,
    config: dict[str, Any],
    prediction_metrics: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """生成图像和视频共用的可追溯运行元数据。"""

    import torch
    import transformers

    # 仓库内输入记录相对路径；外部输入仅记录文件名，避免泄露用户目录。
    try:
        recorded_input_path = input_path.relative_to(project_root).as_posix()
    except ValueError:
        recorded_input_path = input_path.name
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "input": {
            "path": recorded_input_path,
            "sha256": _sha256(input_path),
        },
        "model": {
            "id": config["model"]["id"],
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "weights_sha256": config["model"]["weights_sha256"],
            "road_class_ids": config["segmentation"]["road_class_ids"],
            "road_class_names": config["segmentation"]["road_class_names"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            **prediction_metrics,
        },
    }


def run_image_demo(
    input_path: Path,
    output_directory: Path,
    segmenter: SegformerDrivableAreaSegmenter,
) -> dict[str, Any]:
    """运行单图分割并保存掩码、边界、置信度图、叠加图和结果 JSON。"""

    image = Image.open(input_path).convert("RGB")
    prediction = segmenter.predict(image)
    config = segmenter.config
    visualization = config["visualization"]
    overlay = render_overlay(
        image,
        prediction,
        tuple(visualization["road_color_rgb"]),
        tuple(visualization["boundary_color_rgb"]),
        float(visualization["overlay_alpha"]),
    )

    # 产物名称固定，便于验收脚本和后续模块稳定读取。
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "drivable-mask.png"
    boundary_path = output_directory / "drivable-boundary.png"
    confidence_path = output_directory / "road-confidence.png"
    overlay_path = output_directory / "overlay.png"
    Image.fromarray(prediction.mask.astype(np.uint8) * 255).save(mask_path)
    Image.fromarray(prediction.boundary.astype(np.uint8) * 255).save(boundary_path)
    Image.fromarray(np.clip(prediction.confidence * 255.0, 0, 255).astype(np.uint8)).save(
        confidence_path
    )
    overlay.save(overlay_path, optimize=True)

    metrics = _prediction_metrics(prediction)
    metadata = _base_metadata(input_path, config, metrics, segmenter.project_root)
    metadata["input"].update({"width": image.width, "height": image.height, "type": "image"})
    metadata["outputs"] = {
        "mask": mask_path.name,
        "boundary": boundary_path.name,
        "confidence": confidence_path.name,
        "overlay": overlay_path.name,
    }
    # 使用 list 快照，避免在遍历字典时追加哈希字段导致迭代错误。
    for name, filename in list(metadata["outputs"].items()):
        metadata["outputs"][f"{name}_sha256"] = _sha256(output_directory / filename)
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_video_demo(
    input_path: Path,
    output_directory: Path,
    segmenter: SegformerDrivableAreaSegmenter,
) -> dict[str, Any]:
    """逐帧运行视频分割并输出保持原分辨率和帧率的叠加视频。"""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video input requires opencv-python-headless") from exc

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 极少数容器不报告帧率；此时使用保守的 25 FPS 作为回退值。
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "overlay.mp4"
    codec = cv2.VideoWriter_fourcc(*str(segmenter.config["video"]["output_codec"]))
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot create output video with configured codec")

    frame_metrics: list[dict[str, Any]] = []
    visualization = segmenter.config["visualization"]
    # ``finally`` 保证异常发生时也释放解码器和编码器句柄。
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            prediction = segmenter.predict(image)
            overlay = render_overlay(
                image,
                prediction,
                tuple(visualization["road_color_rgb"]),
                tuple(visualization["boundary_color_rgb"]),
                float(visualization["overlay_alpha"]),
            )
            writer.write(cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR))
            frame_metrics.append(_prediction_metrics(prediction))
    finally:
        capture.release()
        writer.release()
    if not frame_metrics:
        raise RuntimeError("video contains no readable frames")

    # 视频只记录聚合指标，避免长视频产生过大的逐帧 JSON。
    aggregate = {
        "device": frame_metrics[0]["device"],
        "frame_count": len(frame_metrics),
        "average_latency_ms": round(
            sum(float(item["latency_ms"]) for item in frame_metrics) / len(frame_metrics), 3
        ),
        "average_road_coverage": round(
            sum(float(item["road_coverage"]) for item in frame_metrics) / len(frame_metrics), 6
        ),
    }
    metadata = _base_metadata(
        input_path,
        segmenter.config,
        aggregate,
        segmenter.project_root,
    )
    metadata["input"].update(
        {"width": width, "height": height, "fps": fps, "type": "video"}
    )
    metadata["outputs"] = {"overlay": output_path.name, "overlay_sha256": _sha256(output_path)}
    (output_directory / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def run_demo(input_path: Path, output_directory: Path, config_path: Path) -> dict[str, Any]:
    """根据输入扩展名分派图像或视频 Demo，并返回运行元数据。"""

    input_path = input_path.resolve()
    config_path = config_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    # 默认配置位于 ``<root>/configs/models``，向上两级得到仓库根目录。
    project_root = config_path.parents[2]
    segmenter = SegformerDrivableAreaSegmenter(config_path, project_root)
    suffix = input_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return run_image_demo(input_path, output_directory.resolve(), segmenter)
    if suffix in VIDEO_SUFFIXES:
        return run_video_demo(input_path, output_directory.resolve(), segmenter)
    raise ValueError(f"unsupported input extension: {suffix}")
