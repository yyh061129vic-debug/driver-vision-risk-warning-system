"""校验任务 4 样例选择、索引和本地可视化产物。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/task4_samples.yaml"


def validate(check_outputs: bool = True) -> list[str]:
    """检查配置与索引；需要时进一步验证 20 张图和总览图。"""

    errors: list[str] = []
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    samples = [
        (dataset["id"], sample)
        for dataset in config.get("datasets", [])
        for sample in dataset.get("samples", [])
    ]
    sample_ids = [sample["sample_id"] for _, sample in samples]
    minimum = int(config.get("minimum_visualization_count", 20))
    if len(samples) < minimum:
        errors.append(f"expected at least {minimum} configured samples, got {len(samples)}")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("task-4 sample ids must be unique")
    dataset_ids = {dataset_id for dataset_id, _ in samples}
    if dataset_ids != {"lost-and-found", "segment-me-if-you-can"}:
        errors.append(f"unexpected task-4 dataset set: {sorted(dataset_ids)}")

    index_path = ROOT / config["sample_index"]
    if not index_path.is_file():
        errors.append(f"missing sample index: {index_path.relative_to(ROOT)}")
    else:
        records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != len(samples):
            errors.append(f"sample index contains {len(records)} records, expected {len(samples)}")
        for record in records:
            if Path(record["image_path"]).is_absolute():
                errors.append(f"absolute path in sample index: {record['sample_id']}")

    # 元数据模式供常规单元测试使用，不要求本机保留忽略的运行产物。
    if not check_outputs:
        return errors

    output_root = ROOT / config["output_root"]
    summary_path = output_root / "summary.json"
    contact_sheet = output_root / "contact-sheet.png"
    if not summary_path.is_file():
        errors.append("missing task-4 summary.json")
        return errors
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("visualization_count") != len(samples):
        errors.append("summary visualization count does not match configuration")
    if set(summary.get("by_dataset", {})) != dataset_ids:
        errors.append("summary does not cover both adopted datasets")
    for item in summary.get("samples", []):
        if item.get("road_pixels", 0) <= 0:
            errors.append(f"{item.get('sample_id')}: no road pixels")
        if item.get("obstacle_pixels", 0) <= 0:
            errors.append(f"{item.get('sample_id')}: no obstacle pixels")
        path = ROOT / item["visualization_path"]
        if not path.is_file():
            errors.append(f"missing visualization: {item['visualization_path']}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError as exc:
            errors.append(f"invalid visualization {path.name}: {exc}")
    if not contact_sheet.is_file():
        errors.append("missing contact sheet")
    else:
        try:
            with Image.open(contact_sheet) as image:
                image.verify()
        except OSError as exc:
            errors.append(f"invalid contact sheet: {exc}")
    return errors


def main() -> int:
    """解析是否只验元数据，并打印任务 4 验收结果。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    errors = validate(check_outputs=not args.metadata_only)
    if errors:
        print("Task-4 visualization validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "configuration and index" if args.metadata_only else "configuration, index, and outputs"
    print(f"Task-4 visualization validation passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
