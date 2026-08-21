"""统一评估道路分割：道路 mask IoU + 边界 F1/IoU（raw vs 切边后 refined）。

与 team 3 sub SegFormer 同口径：在 BDD100K drivable 真值上按二值道路（road=1）
计算指标。用于给"道路边缘切边"后处理建立可复现标尺。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PIDNET_ROOT = ROOT / "external" / "PIDNet"
if str(PIDNET_ROOT) not in sys.path:
    sys.path.insert(0, str(PIDNET_ROOT))

import models  # type: ignore  # noqa: E402
from configs import config, update_config  # type: ignore  # noqa: E402
from driver_vision_risk.inference.postprocess import apply_edge_cut, enhance_drivable_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate road mask IoU + boundary F1/IoU on BDD100K drivable labels.")
    parser.add_argument(
        "--cfg",
        default=str(PIDNET_ROOT / "configs" / "bdd100k" / "pidnet_small_bdd_semantic_retrain.yaml"),
        help="PIDNet yaml config.",
    )
    parser.add_argument(
        "--model-file",
        default=str(Path("D:/pidnet_semantic_retrain/train/bdd_semantic/pidnet_small_bdd_semantic_retrain/best.pt")),
        help="Trained PIDNet checkpoint path.",
    )
    parser.add_argument("--list-path", default=r"D:\bdd100k\list\bdd100k_drivable\val_pilot.lst", help="Drivable eval list.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate at most N samples (0 = all).")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "road_boundary_eval"),
        help="Output directory.",
    )
    parser.add_argument("--road-class-id", type=int, default=0, help="Road class id in BDD semantic trainIds.")
    parser.add_argument("--boundary-tolerance", type=int, default=3, help="Trimap dilation width for boundary metrics.")
    parser.add_argument(
        "--edge-cut",
        default="none",
        choices=["none", "crf", "snake", "lane_trim", "all"],
        help="Which edge-cut method to enable in postprocessing.",
    )
    parser.add_argument("--save-visual-count", type=int, default=0, help="Export overlays for worst N samples (0 = none).")
    return parser.parse_args()


def load_runtime_config(cfg_path: str):
    class Args:
        cfg = cfg_path
        opts: list[str] = []

    update_config(config, Args())
    return config


def load_model(cfg, model_path: Path) -> torch.nn.Module:
    model = models.pidnet.get_seg_model(cfg, imgnet_pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model_dict = model.state_dict()
    remapped = {}
    for key, value in checkpoint.items():
        mapped_key = key[6:] if key.startswith("model.") else key
        if mapped_key in model_dict and value.shape == model_dict[mapped_key].shape:
            remapped[mapped_key] = value
    model_dict.update(remapped)
    model.load_state_dict(model_dict, strict=False)
    return model.cuda().eval()


def load_postprocess_config(edge_cut: str = "none") -> dict:
    cfg_path = ROOT / "configs" / "models" / "pidnet_s_cityscapes.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    postprocess = dict(cfg["postprocess"])
    # BDD100K semantic 的车辆类别 trainId（car..bicycle）。
    postprocess["vehicle_exclusion"]["vehicle_class_ids"] = [13, 14, 15, 16, 17, 18]
    postprocess["edge_cut"] = {
        "crf_enabled": edge_cut in {"crf", "all"},
        "snake_enabled": edge_cut in {"snake", "all"},
        "lane_trim_enabled": edge_cut in {"lane_trim", "all"},
    }
    return {"postprocess": postprocess, "visualization": cfg["visualization"]}


def parse_list_file(data_root: Path, list_path: str) -> list[tuple[Path, Path]]:
    path = Path(list_path)
    items: list[tuple[Path, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_rel, label_rel = line.strip().split()
        items.append((data_root / image_rel, data_root / label_rel))
    return items


def _kernel(width: int):
    width = max(1, int(width))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))


def boundary_region(mask: np.ndarray, width: int) -> np.ndarray:
    """返回 mask 边界两侧 width 像素内的 trimap 带。"""
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


def predict(cfg, model: torch.nn.Module, image: Image.Image, road_class_id: int):
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = image_array.shape[:2]
    normalized = image_array / 255.0
    normalized -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    normalized /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))).unsqueeze(0).cuda()

    with torch.no_grad():
        prediction = model(tensor)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[cfg.TEST.OUTPUT_INDEX]
        prediction = F.interpolate(
            prediction,
            size=(height, width),
            mode="bilinear",
            align_corners=cfg.MODEL.ALIGN_CORNERS,
        )
    logits = prediction[0]
    probabilities = torch.softmax(logits, dim=0)
    class_map = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    confidence = probabilities[road_class_id].cpu().numpy().astype(np.float32)
    road_mask = class_map == road_class_id
    return image_array, road_mask, confidence, class_map


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(args.cfg)
    model_path = Path(args.model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {model_path}")

    data_root = cfg.DATASET.ROOT
    pairs = parse_list_file(Path(data_root), args.list_path)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"no samples found in {args.list_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg, model_path)
    pp_config = load_postprocess_config(args.edge_cut)
    postprocess_cfg = pp_config["postprocess"]

    variants = ["raw", "refined", "crf", "snake", "lane_trim", "all"]
    tp = {name: 0 for name in variants}
    fp = {name: 0 for name in variants}
    fn = {name: 0 for name in variants}
    boundary = {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0} for name in variants}
    per_sample: list[dict] = []

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        gt = (label > 0) & (label != 255)

        image_array, road_mask, confidence, class_map = predict(cfg, model, image, args.road_class_id)
        enhancement = enhance_drivable_prediction(
            image_array.astype(np.uint8),
            road_mask,
            confidence,
            class_map,
            pp_config,
        )
        refined_mask = np.asarray(enhancement["refined_mask"]).astype(np.bool_)
        lane_lines = enhancement["lane_lines"]
        rgb = image_array.astype(np.uint8)

        masks: dict[str, np.ndarray] = {
            "raw": np.asarray(enhancement["raw_mask"]).astype(np.bool_),
            "refined": refined_mask,
        }
        for method in ["crf", "snake", "lane_trim", "all"]:
            postprocess_cfg["edge_cut"] = {
                "crf_enabled": method in {"crf", "all"},
                "snake_enabled": method in {"snake", "all"},
                "lane_trim_enabled": method in {"lane_trim", "all"},
            }
            cut_mask, _ = apply_edge_cut(refined_mask, confidence, rgb, lane_lines, postprocess_cfg)
            masks[method] = cut_mask.astype(np.bool_)

        sample_record: dict = {"image": str(image_path)}
        for name in variants:
            mask = masks[name]
            tp[name] += int(np.logical_and(mask, gt).sum())
            fp[name] += int(np.logical_and(mask, ~gt).sum())
            fn[name] += int(np.logical_and(~mask, gt).sum())
            bm = boundary_metrics(mask, gt, args.boundary_tolerance)
            for key in boundary[name]:
                boundary[name][key] += bm[key]
            sample_record[f"{name}_road_iou"] = round(binary_iou(mask, gt), 4)
            sample_record[f"{name}_boundary_f1"] = round(bm["f1"], 4)
            sample_record[f"{name}_boundary_iou"] = round(bm["iou"], 4)
        per_sample.append(sample_record)

        if index % 20 == 0 or index == len(pairs):
            print(f"[road-boundary] processed {index}/{len(pairs)}")

    n = len(pairs)
    summary: dict = {
        "model_file": str(model_path),
        "config_file": str(args.cfg),
        "list_path": args.list_path,
        "num_samples": n,
        "edge_cut_method": args.edge_cut,
        "boundary_tolerance": args.boundary_tolerance,
        "road_class_id": args.road_class_id,
    }
    for name in variants:
        union = tp[name] + fp[name] + fn[name]
        precision = tp[name] / (tp[name] + fp[name]) if (tp[name] + fp[name]) else 0.0
        recall = tp[name] / (tp[name] + fn[name]) if (tp[name] + fn[name]) else 0.0
        summary[name] = {
            "road_iou": round(tp[name] / union, 4) if union else 0.0,
            "road_precision": round(precision, 4),
            "road_recall": round(recall, 4),
            "boundary_f1": round(boundary[name]["f1"] / n, 4),
            "boundary_iou": round(boundary[name]["iou"] / n, 4),
            "boundary_precision": round(boundary[name]["precision"] / n, 4),
            "boundary_recall": round(boundary[name]["recall"] / n, 4),
        }
    summary["delta_vs_raw"] = {
        name: {
            "road_iou": round(summary[name]["road_iou"] - summary["raw"]["road_iou"], 4),
            "boundary_f1": round(summary[name]["boundary_f1"] - summary["raw"]["boundary_f1"], 4),
            "boundary_iou": round(summary[name]["boundary_iou"] - summary["raw"]["boundary_iou"], 4),
        }
        for name in variants
        if name != "raw"
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "per_sample.json").write_text(json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
