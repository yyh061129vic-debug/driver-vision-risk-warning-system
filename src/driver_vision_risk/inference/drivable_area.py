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
    binary_dilate,
    binary_erode,
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


def render_anomaly_heatmap(
    anomaly: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[Image.Image, dict[str, float]]:
    """把原始异常分数稳健归一化为蓝到红的可视化热力图。"""

    if anomaly.ndim != 2 or not np.isfinite(anomaly).all():
        raise ValueError("anomaly must be a finite two-dimensional array")
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError("heatmap percentiles must satisfy 0 <= low < high <= 100")
    lower = float(np.percentile(anomaly, percentile_low))
    upper = float(np.percentile(anomaly, percentile_high))
    # 恒定分数图避免除零；此时整张图映射到最低可疑颜色。
    if upper <= lower:
        normalized = np.zeros_like(anomaly, dtype=np.float32)
    else:
        normalized = np.clip((anomaly - lower) / (upper - lower), 0.0, 1.0)

    # 使用紧凑的蓝—青—黄—红色带，不新增 Matplotlib 运行依赖。
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    heatmap = np.stack((red, green, blue), axis=-1)
    image = Image.fromarray(np.rint(heatmap * 255.0).astype(np.uint8), mode="RGB")
    return image, {
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
        "score_low": round(lower, 6),
        "score_high": round(upper, 6),
    }


def categorical_boundary_band(class_map: np.ndarray, dilation_pixels: int) -> np.ndarray:
    """检测 argmax 类别交界线，并膨胀为需要从异常候选中排除的边界带。"""

    if class_map.ndim != 2:
        raise ValueError("class map must be a two-dimensional array")
    if dilation_pixels < 0:
        raise ValueError("class boundary dilation pixels must not be negative")
    boundary = np.zeros(class_map.shape, dtype=np.bool_)
    # 相邻类别不同时把交界线两侧都标记，避免只偏向其中一个语义类别。
    horizontal_difference = class_map[:, 1:] != class_map[:, :-1]
    boundary[:, 1:] |= horizontal_difference
    boundary[:, :-1] |= horizontal_difference
    vertical_difference = class_map[1:, :] != class_map[:-1, :]
    boundary[1:, :] |= vertical_difference
    boundary[:-1, :] |= vertical_difference
    return binary_dilate(boundary, width=dilation_pixels)


def anomaly_eligible_region(
    prediction: DrivableAreaPrediction,
    road_mask_erosion_pixels: int,
    class_boundary_suppression_pixels: int,
) -> np.ndarray:
    """生成道路内部且不属于类别边界带的异常分析有效区域。"""

    if prediction.anomaly.shape != prediction.mask.shape:
        raise ValueError("anomaly score shape must match the drivable-area mask")
    if prediction.class_map.shape != prediction.mask.shape:
        raise ValueError("class map shape must match the drivable-area mask")
    if road_mask_erosion_pixels < 0:
        raise ValueError("road mask erosion pixels must not be negative")
    if class_boundary_suppression_pixels < 0:
        raise ValueError("class boundary suppression pixels must not be negative")

    road_region = binary_erode(prediction.mask, width=road_mask_erosion_pixels)
    if class_boundary_suppression_pixels:
        boundary_band = categorical_boundary_band(
            prediction.class_map,
            dilation_pixels=class_boundary_suppression_pixels,
        )
    else:
        boundary_band = np.zeros(prediction.mask.shape, dtype=np.bool_)
    return road_region & ~boundary_band


def mask_anomaly_heatmap(heatmap: Image.Image, eligible_region: np.ndarray) -> Image.Image:
    """把道路 ROI 与类别边界带之外的热力图像素置黑，仅保留风险分析区域。"""

    heatmap_array = np.asarray(heatmap.convert("RGB")).copy()
    if eligible_region.shape != heatmap_array.shape[:2]:
        raise ValueError("eligible anomaly region shape must match the heatmap")
    heatmap_array[~eligible_region] = 0
    return Image.fromarray(heatmap_array, mode="RGB")


def extract_anomaly_regions(
    prediction: DrivableAreaPrediction,
    threshold: float,
    minimum_area_pixels: int,
    connectivity: int = 8,
    road_mask_erosion_pixels: int = 0,
    class_boundary_suppression_pixels: int = 0,
) -> list[dict[str, Any]]:
    """从腐蚀后的预测道路内部提取超过阈值的连通异常区域。"""

    if minimum_area_pixels < 1:
        raise ValueError("minimum anomaly component area must be at least one pixel")
    if connectivity not in {4, 8}:
        raise ValueError("anomaly component connectivity must be either 4 or 8")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("anomaly region extraction requires opencv-python-headless") from exc

    # 先收缩道路 ROI，再排除类别边界带；路外背景与天然犹豫边界均不参与连通域。
    eligible_region = anomaly_eligible_region(
        prediction,
        road_mask_erosion_pixels,
        class_boundary_suppression_pixels,
    )
    # 不直接把 Energy 写成零，因为零可能高于负阈值；布尔排除对 MSP 和 Energy 都安全。
    candidate_mask = eligible_region & (prediction.anomaly > float(threshold))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=connectivity,
    )
    regions: list[dict[str, Any]] = []
    for label_id in range(1, component_count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < minimum_area_pixels:
            continue
        component_scores = prediction.anomaly[labels == label_id]
        regions.append(
            {
                # 右下坐标采用 Python 切片习惯的开区间，便于状态机直接裁剪图像。
                "bbox_xyxy": [x, y, x + width, y + height],
                "area_pixels": area,
                "mean_anomaly_score": round(float(component_scores.mean()), 6),
            }
        )
    # 状态机优先读取最可疑区域，因此按平均异常分数从高到低排列。
    regions.sort(key=lambda region: float(region["mean_anomaly_score"]), reverse=True)
    return regions


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
            "model_input_size": config.get("runtime", {}).get(
                "input_size", "checkpoint_default"
            ),
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
    anomaly_config = config.get("anomaly", {})
    raw_heatmap, heatmap_scale = render_anomaly_heatmap(
        prediction.anomaly,
        float(anomaly_config.get("heatmap_percentile_low", 2.0)),
        float(anomaly_config.get("heatmap_percentile_high", 98.0)),
    )
    threshold = anomaly_config.get("threshold")
    minimum_area_pixels = int(anomaly_config.get("minimum_area_pixels", 1))
    connectivity = int(anomaly_config.get("connectivity", 8))
    road_mask_erosion_pixels = int(anomaly_config.get("road_mask_erosion_pixels", 0))
    class_boundary_suppression_pixels = int(
        anomaly_config.get("class_boundary_suppression_pixels", 0)
    )
    eligible_region = anomaly_eligible_region(
        prediction,
        road_mask_erosion_pixels,
        class_boundary_suppression_pixels,
    )
    # 风险热力图只显示真正参与候选提取的区域，原始热力图另存供模型诊断。
    filtered_heatmap = mask_anomaly_heatmap(raw_heatmap, eligible_region)
    regions = (
        extract_anomaly_regions(
            prediction,
            float(threshold),
            minimum_area_pixels,
            connectivity,
            road_mask_erosion_pixels,
            class_boundary_suppression_pixels,
        )
        if threshold is not None
        else []
    )

    # 产物名称固定，便于验收脚本和后续模块稳定读取。
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "drivable-mask.png"
    boundary_path = output_directory / "drivable-boundary.png"
    confidence_path = output_directory / "road-confidence.png"
    anomaly_raw_heatmap_path = output_directory / "anomaly-raw-heatmap.png"
    anomaly_heatmap_path = output_directory / "anomaly-heatmap.png"
    overlay_path = output_directory / "overlay.png"
    Image.fromarray(prediction.mask.astype(np.uint8) * 255).save(mask_path)
    Image.fromarray(prediction.boundary.astype(np.uint8) * 255).save(boundary_path)
    Image.fromarray(np.clip(prediction.confidence * 255.0, 0, 255).astype(np.uint8)).save(
        confidence_path
    )
    raw_heatmap.save(anomaly_raw_heatmap_path, optimize=True)
    filtered_heatmap.save(anomaly_heatmap_path, optimize=True)
    overlay.save(overlay_path, optimize=True)

    metrics = _prediction_metrics(prediction)
    metadata = _base_metadata(input_path, config, metrics, segmenter.project_root)
    metadata["input"].update({"width": image.width, "height": image.height, "type": "image"})
    metadata["anomaly_detection"] = {
        "score_method": segmenter.anomaly_score_method,
        "score_direction": "higher_is_more_suspicious",
        "candidate_rule": (
            "eroded_predicted_road_excluding_class_boundary_band_"
            "and_score_above_threshold"
        ),
        "threshold": threshold,
        "threshold_status": anomaly_config.get("threshold_status", "configured"),
        "minimum_area_pixels": minimum_area_pixels,
        "connectivity": connectivity,
        "road_mask_erosion_pixels": road_mask_erosion_pixels,
        "class_boundary_suppression_pixels": class_boundary_suppression_pixels,
        "class_boundary_suppression_mode": "candidate_mask_exclusion",
        "heatmap_output": "road_roi_with_class_boundary_band_suppressed",
        "heatmap_scale": heatmap_scale,
        "region_count": len(regions),
        "regions": regions,
    }
    metadata["outputs"] = {
        "mask": mask_path.name,
        "boundary": boundary_path.name,
        "confidence": confidence_path.name,
        "anomaly_raw_heatmap": anomaly_raw_heatmap_path.name,
        "anomaly_heatmap": anomaly_heatmap_path.name,
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
