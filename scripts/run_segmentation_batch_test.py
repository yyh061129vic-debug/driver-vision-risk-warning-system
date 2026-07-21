"""使用固定 SegFormer 模型批量测试 20 张已登记道路图片。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from driver_vision_risk.inference.drivable_area import run_image_demo
from driver_vision_risk.models.segformer import (
    SegformerDrivableAreaSegmenter,
    binary_inner_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/indexes/task4_samples.jsonl"
DEFAULT_DATA_CONFIG = ROOT / "configs/data/task4_samples.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "configs/models/segformer_cityscapes.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/task5_segmentation_20_images/gpu"
COUNT_KEYS = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "pred_boundary_count",
    "gt_boundary_count",
    "pred_boundary_hits",
    "gt_boundary_hits",
)


def _font(size: int) -> ImageFont.ImageFont:
    """选择可用字体；找不到系统字体时回退到 Pillow 默认字体。"""

    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _load_samples(index_path: Path, limit: int) -> list[dict[str, Any]]:
    """按索引原有顺序读取指定数量的样本，保证测试选择可复现。"""

    samples: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) == limit:
                break
    if len(samples) != limit:
        raise ValueError(f"sample index contains {len(samples)} records, expected {limit}")
    return samples


def _dilate_four_connected(mask: np.ndarray, radius: int) -> np.ndarray:
    """用四邻域膨胀边界，为边界 F-score 提供像素容差区域。"""

    dilated = mask.copy()
    for _ in range(radius):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return dilated


def _divide(numerator: int, denominator: int) -> float:
    """安全计算比例；当前测试不存在有效像素时返回零。"""

    return numerator / denominator if denominator else 0.0


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """从累计混淆计数生成冻结方案规定的道路与边界指标。"""

    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    true_negative = counts["true_negative"]
    road_iou = _divide(
        true_positive,
        true_positive + false_positive + false_negative,
    )
    nonroad_iou = _divide(
        true_negative,
        true_negative + false_positive + false_negative,
    )
    boundary_precision = _divide(
        counts["pred_boundary_hits"],
        counts["pred_boundary_count"],
    )
    boundary_recall = _divide(
        counts["gt_boundary_hits"],
        counts["gt_boundary_count"],
    )
    boundary_fscore = _divide(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )
    # 百分比字段都带 ``_pct`` 后缀，避免与 Demo 中的 0 到 1 比例混淆。
    return {
        "road_precision_pct": round(
            100.0 * _divide(true_positive, true_positive + false_positive), 3
        ),
        "road_recall_pct": round(
            100.0 * _divide(true_positive, true_positive + false_negative), 3
        ),
        "road_dice_pct": round(
            100.0
            * _divide(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            ),
            3,
        ),
        "road_iou_pct": round(100.0 * road_iou, 3),
        "nonroad_iou_pct": round(100.0 * nonroad_iou, 3),
        "binary_miou_pct": round(100.0 * (road_iou + nonroad_iou) / 2.0, 3),
        "boundary_fscore_pct": round(100.0 * boundary_fscore, 3),
        # 两个数据集的有效非道路像素均为已标注障碍物，因此可直接报告风险相关诊断。
        "obstacle_rejection_pct": round(
            100.0 * _divide(true_negative, true_negative + false_positive), 3
        ),
        "obstacle_predicted_as_road_pct": round(
            100.0 * _divide(false_positive, true_negative + false_positive), 3
        ),
    }


def _evaluate_prediction(
    prediction_mask: np.ndarray,
    annotation: np.ndarray,
    label_mapping: dict[str, Any],
    boundary_tolerance: int = 3,
) -> tuple[dict[str, int], dict[str, float]]:
    """把模型道路掩码与数据集真值统一后计算单图指标。"""

    if prediction_mask.shape != annotation.shape:
        raise ValueError(
            f"prediction shape {prediction_mask.shape} does not match annotation {annotation.shape}"
        )
    void_values = label_mapping.get("void_values", [])
    valid = ~np.isin(annotation, void_values)
    ground_truth = np.isin(annotation, label_mapping["road_values"]) & valid
    prediction = prediction_mask.astype(np.bool_) & valid

    # 混淆矩阵只统计非 void 像素，保持与冻结实验方案一致。
    counts = {
        "true_positive": int((prediction & ground_truth).sum()),
        "false_positive": int((prediction & ~ground_truth & valid).sum()),
        "false_negative": int((~prediction & ground_truth).sum()),
        "true_negative": int((~prediction & ~ground_truth & valid).sum()),
    }

    # 边界匹配采用四邻域和 3 像素对称容差，口径来自冻结方案。
    pred_boundary = binary_inner_boundary(prediction)
    gt_boundary = binary_inner_boundary(ground_truth)
    pred_match_region = _dilate_four_connected(pred_boundary, boundary_tolerance)
    gt_match_region = _dilate_four_connected(gt_boundary, boundary_tolerance)
    counts.update(
        {
            "pred_boundary_count": int(pred_boundary.sum()),
            "gt_boundary_count": int(gt_boundary.sum()),
            "pred_boundary_hits": int((pred_boundary & gt_match_region).sum()),
            "gt_boundary_hits": int((gt_boundary & pred_match_region).sum()),
        }
    )
    return counts, _metrics_from_counts(counts)


def _contact_sheet(items: list[tuple[str, Image.Image]], destination: Path) -> None:
    """把 20 张模型叠加结果排成四列总览图，便于组内快速检查。"""

    columns = 4
    thumbnail_width, thumbnail_height, label_height = 320, 180, 30
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        "#111827",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (sample_id, image) in enumerate(items):
        row, column = divmod(index, columns)
        thumbnail = image.copy()
        thumbnail.thumbnail((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)
        x = column * thumbnail_width + (thumbnail_width - thumbnail.width) // 2
        y = row * (thumbnail_height + label_height) + (thumbnail_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (
                column * thumbnail_width + 8,
                row * (thumbnail_height + label_height) + thumbnail_height + 6,
            ),
            sample_id,
            fill="#e5e7eb",
            font=_font(14),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)


def main() -> int:
    """加载一次模型，识别 20 张图片并写出逐图产物和汇总结果。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be at least one")

    samples = _load_samples(args.index.resolve(), args.count)
    data_config = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    mappings = {
        str(dataset["id"]): dataset["label_mapping"]
        for dataset in data_config["datasets"]
    }
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 模型只加载一次，连续完成全部图片推理，避免重复读取 208 个权重分片。
    segmenter = SegformerDrivableAreaSegmenter(args.model_config.resolve(), ROOT)
    dataset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {key: 0 for key in COUNT_KEYS}
    )
    dataset_sample_counts: dict[str, int] = defaultdict(int)
    dataset_anomaly_counts: dict[str, int] = defaultdict(int)
    sample_results: list[dict[str, Any]] = []
    contact_items: list[tuple[str, Image.Image]] = []
    anomaly_contact_items: list[tuple[str, Image.Image]] = []
    latencies: list[float] = []

    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        dataset_id = str(sample["dataset_id"])
        image_path = ROOT / sample["image_path"]
        annotation_path = ROOT / sample["annotation_paths"]["semantic_mask"]
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(f"missing image or annotation for {sample_id}")

        sample_output = output_root / "samples" / sample_id
        result = run_image_demo(image_path, sample_output, segmenter)
        prediction_mask = np.asarray(Image.open(sample_output / "drivable-mask.png")) > 0
        annotation = np.asarray(Image.open(annotation_path))
        counts, metrics = _evaluate_prediction(
            prediction_mask,
            annotation,
            mappings[dataset_id],
        )
        for key in COUNT_KEYS:
            dataset_counts[dataset_id][key] += counts[key]
        dataset_sample_counts[dataset_id] += 1

        runtime = result["runtime"]
        anomaly_detection = result["anomaly_detection"]
        anomaly_region_count = int(anomaly_detection["region_count"])
        dataset_anomaly_counts[dataset_id] += anomaly_region_count
        latency_ms = float(runtime["latency_ms"])
        latencies.append(latency_ms)
        sample_results.append(
            {
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "input_path": sample["image_path"],
                "output_directory": sample_output.relative_to(ROOT).as_posix(),
                "latency_ms": latency_ms,
                "road_coverage_pct": round(100.0 * float(runtime["road_coverage"]), 3),
                "mean_road_confidence": float(runtime["mean_road_confidence"]),
                "anomaly_region_count": anomaly_region_count,
                "confusion_counts": counts,
                "metrics": metrics,
            }
        )
        contact_items.append((sample_id, Image.open(sample_output / "overlay.png").convert("RGB")))
        anomaly_contact_items.append(
            (sample_id, Image.open(sample_output / "anomaly-heatmap.png").convert("RGB"))
        )
        print(
            f"[{index:02d}/{len(samples):02d}] {sample_id}: "
            f"{latency_ms:.3f} ms, road IoU {metrics['road_iou_pct']:.3f}%, "
            f"anomaly regions {anomaly_region_count}"
        )

    contact_sheet = output_root / "contact-sheet.png"
    anomaly_contact_sheet = output_root / "anomaly-contact-sheet.png"
    _contact_sheet(contact_items, contact_sheet)
    _contact_sheet(anomaly_contact_items, anomaly_contact_sheet)
    dataset_results = {
        dataset_id: {
            "sample_count": dataset_sample_counts[dataset_id],
            "confusion_counts": counts,
            "metrics": _metrics_from_counts(counts),
        }
        for dataset_id, counts in sorted(dataset_counts.items())
    }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task": "segmentation_20_image_test",
        "sample_count": len(samples),
        "selection": {
            "index": args.index.resolve().relative_to(ROOT).as_posix(),
            "order": "first_n_in_registered_index",
        },
        "model": {
            "id": segmenter.config["model"]["id"],
            "revision": segmenter.config["model"]["revision"],
            "device": str(segmenter.device),
        },
        # 本次是交互式 GPU 批量测试，不冒充冻结方案中的 CPU 性能基线。
        "performance": {
            "protocol": "interactive_gpu_batch_not_frozen_cpu_baseline",
            "latency_mean_ms": round(statistics.fmean(latencies), 3),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "throughput_fps": round(1000.0 / statistics.fmean(latencies), 3),
        },
        "anomaly_detection": {
            "score_method": segmenter.anomaly_score_method,
            "threshold": segmenter.config.get("anomaly", {}).get("threshold"),
            "minimum_area_pixels": segmenter.config.get("anomaly", {}).get(
                "minimum_area_pixels"
            ),
            "road_mask_erosion_pixels": segmenter.config.get("anomaly", {}).get(
                "road_mask_erosion_pixels", 0
            ),
            "class_boundary_suppression_pixels": segmenter.config.get("anomaly", {}).get(
                "class_boundary_suppression_pixels", 0
            ),
            "total_region_count": sum(dataset_anomaly_counts.values()),
            "region_count_by_dataset": dict(sorted(dataset_anomaly_counts.items())),
        },
        "evaluation": {
            "aggregation": "per_dataset_confusion_accumulation",
            "cross_dataset_headline_pooling": False,
            "boundary_tolerance_pixels": 3,
            "datasets": dataset_results,
        },
        "outputs": {
            "contact_sheet": contact_sheet.relative_to(ROOT).as_posix(),
            "anomaly_contact_sheet": anomaly_contact_sheet.relative_to(ROOT).as_posix(),
            "sample_directory": (output_root / "samples").relative_to(ROOT).as_posix(),
        },
        "samples": sample_results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "sample_count",
                    "model",
                    "performance",
                    "anomaly_detection",
                    "evaluation",
                    "outputs",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
