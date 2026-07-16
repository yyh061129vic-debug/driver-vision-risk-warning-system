"""生成任务 4 的道路与障碍物标注叠加图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/task4_samples.yaml"


def _font(size: int) -> ImageFont.ImageFont:
    """按 Windows、Linux 顺序选择字体，均不可用时回退到 Pillow 默认字体。"""

    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _sha256(path: Path) -> str:
    """分块计算输入图像或标注文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_paths(dataset_id: str, sample: dict[str, object]) -> tuple[Path, Path]:
    """根据数据集目录结构解析原图和语义标注路径。"""

    if dataset_id == "lost-and-found":
        split = str(sample["split"])
        sequence = str(sample["sequence_id"])
        frame = str(sample["frame_id"])
        stem = f"{sequence}_{frame}"
        image = (
            ROOT
            / "data_raw/lost-and-found/leftImg8bit_samples/leftImg8bit"
            / split
            / sequence
            / f"{stem}_leftImg8bit.png"
        )
        annotation = (
            ROOT
            / "data_raw/lost-and-found/gtCoarse/gtCoarse"
            / split
            / sequence
            / f"{stem}_gtCoarse_labelIds.png"
        )
        return image, annotation
    if dataset_id == "segment-me-if-you-can":
        scene = str(sample["scene_id"])
        base = (
            ROOT
            / "data_raw/segment-me-if-you-can/road-obstacle-21/extracted/dataset_ObstacleTrack"
        )
        return (
            base / "images" / f"{scene}.webp",
            base / "labels_masks" / f"{scene}_labels_semantic.png",
        )
    raise ValueError(f"unsupported dataset: {dataset_id}")


def _masks(annotation: np.ndarray, mapping: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """按数据集标签映射生成道路掩码和障碍物掩码。"""

    road = np.isin(annotation, mapping["road_values"])
    if "obstacle_values" in mapping:
        obstacle = np.isin(annotation, mapping["obstacle_values"])
    else:
        lower = int(mapping["obstacle_min_value"])
        upper = int(mapping["obstacle_max_value"])
        obstacle = (annotation >= lower) & (annotation <= upper)
    return road, obstacle


def _edge(mask: np.ndarray) -> np.ndarray:
    """通过一次四邻域腐蚀计算障碍物的单像素内边界。"""

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def _blend(base: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    """在指定掩码区域执行原地 alpha 混合。"""

    base[mask] = base[mask] * (1.0 - alpha) + color * alpha


def render_overlay(
    image: Image.Image,
    annotation: np.ndarray,
    mapping: dict[str, object],
    palette: dict[str, object],
    title: str,
) -> tuple[Image.Image, int, int]:
    """绘制单个样例叠加图，并返回道路及障碍物像素数量。"""

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if annotation.shape != image_array.shape[:2]:
        raise ValueError(f"annotation shape {annotation.shape} does not match image {image_array.shape[:2]}")
    # 不同数据集先统一为两个布尔掩码，之后共享同一套着色逻辑。
    road, obstacle = _masks(annotation, mapping)
    _blend(
        image_array,
        road,
        np.asarray(palette["road_rgb"], dtype=np.float32),
        float(palette["road_alpha"]),
    )
    _blend(
        image_array,
        obstacle,
        np.asarray(palette["obstacle_rgb"], dtype=np.float32),
        float(palette["obstacle_alpha"]),
    )
    image_array[_edge(obstacle)] = np.asarray(palette["obstacle_edge_rgb"], dtype=np.float32)
    overlay = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB")

    # 输出图限制最大宽度，兼顾人工检查清晰度和产物体积。
    max_width = 1280
    if overlay.width > max_width:
        height = round(overlay.height * max_width / overlay.width)
        overlay = overlay.resize((max_width, height), Image.Resampling.LANCZOS)

    header_height = 54
    footer_height = 34
    canvas = Image.new("RGB", (overlay.width, overlay.height + header_height + footer_height), "#111827")
    canvas.paste(overlay, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 15), title, fill="white", font=_font(22))
    draw.rectangle((18, canvas.height - 25, 36, canvas.height - 9), fill=tuple(palette["road_rgb"]))
    draw.text((43, canvas.height - 27), "Road / drivable", fill="#d1d5db", font=_font(15))
    draw.rectangle((185, canvas.height - 25, 203, canvas.height - 9), fill=tuple(palette["obstacle_rgb"]))
    draw.text((210, canvas.height - 27), "Obstacle / anomaly", fill="#d1d5db", font=_font(15))
    return canvas, int(road.sum()), int(obstacle.sum())


def _write_index(records: list[dict[str, object]], path: Path) -> None:
    """以 JSON Lines 格式写出逐样例索引，便于流式读取。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _contact_sheet(items: list[tuple[str, Image.Image]], destination: Path) -> None:
    """把全部样例缩略图排成四列总览图，方便一次性人工审阅。"""

    columns = 4
    thumb_width, thumb_height, label_height = 320, 180, 30
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#111827")
    draw = ImageDraw.Draw(sheet)
    for index, (sample_id, image) in enumerate(items):
        row, column = divmod(index, columns)
        thumbnail = image.copy()
        thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = column * thumb_width + (thumb_width - thumbnail.width) // 2
        y = row * (thumb_height + label_height) + (thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (column * thumb_width + 8, row * (thumb_height + label_height) + thumb_height + 6),
            sample_id,
            fill="#e5e7eb",
            font=_font(14),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)


def main() -> int:
    """读取样例配置，生成 20 张叠加图、索引、摘要和总览图。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    sample_output = output_root / "samples"
    sample_output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    summary_samples: list[dict[str, object]] = []
    contact_items: list[tuple[str, Image.Image]] = []
    # 逐数据集、逐样例处理，任何原图或标注缺失都立即终止。
    for dataset in config["datasets"]:
        dataset_id = str(dataset["id"])
        for sample in dataset["samples"]:
            image_path, annotation_path = _sample_paths(dataset_id, sample)
            if not image_path.is_file() or not annotation_path.is_file():
                raise FileNotFoundError(
                    f"missing raw sample: {image_path.relative_to(ROOT)} or {annotation_path.relative_to(ROOT)}"
                )
            image = Image.open(image_path)
            annotation = np.asarray(Image.open(annotation_path))
            title = f"{dataset['display_name']} | {sample['sample_id']} | {sample['split']}"
            overlay, road_pixels, obstacle_pixels = render_overlay(
                image,
                annotation,
                dataset["label_mapping"],
                config["palette"],
                title,
            )
            output_path = sample_output / f"{sample['sample_id']}.png"
            overlay.save(output_path, optimize=True)
            contact_items.append((str(sample["sample_id"]), overlay))

            record = {
                "sample_id": sample["sample_id"],
                "dataset_id": dataset_id,
                "split": sample["split"],
                "sequence_id": sample.get("sequence_id"),
                "scene_id": sample.get("scene_id"),
                "frame_id": sample.get("frame_id"),
                "timestamp": None,
                "camera_id": "left" if dataset_id == "lost-and-found" else None,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "annotation_paths": {"semantic_mask": annotation_path.relative_to(ROOT).as_posix()},
                "metadata": {"usage": dataset["usage"]},
            }
            records.append(record)
            summary_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset_id": dataset_id,
                    "road_pixels": road_pixels,
                    "obstacle_pixels": obstacle_pixels,
                    "image_sha256": _sha256(image_path),
                    "annotation_sha256": _sha256(annotation_path),
                    "visualization_path": output_path.relative_to(ROOT).as_posix(),
                }
            )
            print(f"Rendered {output_path.relative_to(ROOT)}")

    _write_index(records, ROOT / config["sample_index"])
    contact_sheet = output_root / "contact-sheet.png"
    _contact_sheet(contact_items, contact_sheet)
    # 摘要保留数据集分布与输入哈希，支持验收和后续复现。
    by_dataset: dict[str, int] = {}
    for sample in summary_samples:
        dataset_id = str(sample["dataset_id"])
        by_dataset[dataset_id] = by_dataset.get(dataset_id, 0) + 1
    summary = {
        "schema_version": 1,
        "task": config["task"],
        "visualization_count": len(summary_samples),
        "by_dataset": by_dataset,
        "contact_sheet": contact_sheet.relative_to(ROOT).as_posix(),
        "samples": summary_samples,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(summary_samples)} task-4 visualizations and one contact sheet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
