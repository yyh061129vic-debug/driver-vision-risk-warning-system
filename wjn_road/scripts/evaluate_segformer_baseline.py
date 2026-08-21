"""Evaluate team 3 sub road SegFormer baseline on BDD100K drivable val.

与 `evaluate_road_boundary.py` 同口径：在同一批 BDD100K drivable 真值上按
二值道路（road=1）计算道路 mask IoU（全局像素 IoU + 平均逐图 IoU）与边界
F1/IoU。直接加载 team 3 sub 的 road SegFormer-b2 checkpoint，复刻其
`_load_segformer` / `_predict_road_candidates` 的前处理与推理逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate team 3 sub SegFormer road baseline.")
    parser.add_argument(
        "--checkpoint",
        default=r"c:\Users\wangjianing\Documents\trae_projects\csi_intern\temp\driver-vision-risk-warning-system-team3sub\team 3 sub\林展毅_road_segformer_b2_bdd_best.pt",
        help="team 3 sub road SegFormer checkpoint.",
    )
    parser.add_argument("--data-root", default=r"D:\bdd100k", help="BDD100K root.")
    parser.add_argument("--list-path", default=r"D:\bdd100k\list\bdd100k_drivable\val_pilot.lst", help="Drivable eval list.")
    parser.add_argument("--max-samples", type=int, default=100, help="Evaluate at most N samples (0 = all).")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "segformer_baseline_eval"), help="Output dir.")
    parser.add_argument("--boundary-tolerance", type=int, default=3, help="Trimap dilation width for boundary metrics.")
    parser.add_argument("--inference-height", type=int, default=384)
    parser.add_argument("--inference-width", type=int, default=672)
    return parser.parse_args()


def parse_list_file(data_root: Path, list_path: str) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []
    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_rel, label_rel = line.strip().split()
        items.append((data_root / image_rel, data_root / label_rel))
    return items


def _kernel(width: int):
    width = max(1, int(width))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))


def boundary_region(mask: np.ndarray, width: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = _kernel(width)
    dilated = cv2.dilate(mask_u8, kernel) > 0
    eroded = cv2.erode(mask_u8, kernel) > 0
    return dilated & ~eroded


def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(np.bool_)
    target_b = target.astype(np.bool_)
    inter = int(np.logical_and(pred_b, target_b).sum())
    union = int(np.logical_or(pred_b, target_b).sum())
    return inter / union if union else 0.0


def boundary_metrics(pred: np.ndarray, target: np.ndarray, width: int) -> dict[str, float]:
    pred_b = boundary_region(pred, width)
    target_b = boundary_region(target, width)
    inter = int(np.logical_and(pred_b, target_b).sum())
    pred_sum = int(pred_b.sum())
    target_sum = int(target_b.sum())
    precision = inter / pred_sum if pred_sum else 0.0
    recall = inter / target_sum if target_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = int(np.logical_or(pred_b, target_b).sum())
    iou = inter / union if union else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    state_dict = checkpoint.get("state_dict", checkpoint)
    road_class_id = int(checkpoint.get("road_class_id", 1))

    config = SegformerConfig.from_pretrained(model_name)
    config.num_labels = 2
    model = SegformerForSemanticSegmentation(config)
    model.decode_head.classifier = torch.nn.Conv2d(
        model.config.decoder_hidden_size, 2, kernel_size=1
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(
        model_name,
        size={"height": 384, "width": 672},
    )
    return model, processor, road_class_id, model_name


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    model, processor, road_class_id, model_name = load_model(checkpoint_path, device)

    pairs = parse_list_file(Path(args.data_root), args.list_path)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    thresholds = [0.5, 0.2]
    names = ["argmax_0.5", "prod_0.2"]
    tp = {name: 0 for name in names}
    fp = {name: 0 for name in names}
    fn = {name: 0 for name in names}
    boundary = {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0} for name in names}
    per_sample_iou = {name: [] for name in names}
    per_sample: list[dict] = []

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        gt = (label > 0) & (label != 255)

        rgb = np.asarray(image, dtype=np.uint8)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        probability = torch.softmax(logits, dim=1)[0, road_class_id].cpu().numpy()
        height, width = gt.shape

        record = {"image": str(image_path)}
        for name, thr in zip(names, thresholds):
            mask_low = (probability >= thr).astype(np.uint8) * 255
            mask = cv2.resize(mask_low, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            tp[name] += int(np.logical_and(mask, gt).sum())
            fp[name] += int(np.logical_and(mask, ~gt).sum())
            fn[name] += int(np.logical_and(~mask, gt).sum())
            bm = boundary_metrics(mask, gt, args.boundary_tolerance)
            for key in boundary[name]:
                boundary[name][key] += bm[key]
            iou = binary_iou(mask, gt)
            per_sample_iou[name].append(iou)
            record[f"{name}_road_iou"] = round(iou, 4)
            record[f"{name}_boundary_f1"] = round(bm["f1"], 4)
        per_sample.append(record)

        if index % 20 == 0 or index == len(pairs):
            print(f"[segformer-baseline] processed {index}/{len(pairs)}")

    n = len(pairs)
    summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "road_class_id": road_class_id,
        "list_path": args.list_path,
        "num_samples": n,
        "boundary_tolerance": args.boundary_tolerance,
    }
    for name in names:
        union = tp[name] + fp[name] + fn[name]
        precision = tp[name] / (tp[name] + fp[name]) if (tp[name] + fp[name]) else 0.0
        recall = tp[name] / (tp[name] + fn[name]) if (tp[name] + fn[name]) else 0.0
        summary[name] = {
            "road_iou_global": round(tp[name] / union, 4) if union else 0.0,
            "road_iou_mean_per_image": round(float(np.mean(per_sample_iou[name])), 4),
            "road_precision": round(precision, 4),
            "road_recall": round(recall, 4),
            "boundary_f1": round(boundary[name]["f1"] / n, 4),
            "boundary_iou": round(boundary[name]["iou"] / n, 4),
            "boundary_precision": round(boundary[name]["precision"] / n, 4),
            "boundary_recall": round(boundary[name]["recall"] / n, 4),
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "per_sample.json").write_text(json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved artifacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
