"""校验第一版实验方案的冻结字段、数据划分和本地可执行性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "configs/experiments/drivable_area_v1.yaml"
SPLIT_PATH = ROOT / "data/indexes/drivable_area_v1_split.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/models/segformer_cityscapes_cpu.yaml"
TASK4_INDEX_PATH = ROOT / "data/indexes/task4_samples.jsonl"
REQUIRED_PRIMARY_METRICS = {"binary_miou", "road_iou", "boundary_fscore"}


def _load_yaml(path: Path) -> dict[str, object]:
    """读取 YAML 根映射；格式错误或根类型错误时直接抛出异常。"""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path.relative_to(ROOT)}")
    return value


def _task4_records() -> dict[str, dict[str, object]]:
    """按 sample_id 加载任务四索引，供冻结划分复用本地相对路径。"""

    records: dict[str, dict[str, object]] = {}
    for line in TASK4_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[str(record["sample_id"])] = record
    return records


def _validate_frozen_metadata(
    experiment: dict[str, object],
    split: dict[str, object],
    errors: list[str],
) -> None:
    """检查版本、冻结状态、引用路径、随机种子和未确认验收门槛。"""

    metadata = experiment.get("experiment")
    if not isinstance(metadata, dict):
        errors.append("experiment metadata must be a mapping")
    else:
        if metadata.get("id") != "phase1_drivable_area_v1":
            errors.append("experiment id must remain phase1_drivable_area_v1")
        if metadata.get("version") != "1.0.0" or metadata.get("status") != "frozen":
            errors.append("experiment version 1.0.0 must remain frozen")
        if not metadata.get("frozen_on"):
            errors.append("experiment frozen_on is required")

    if split.get("status") != "frozen" or split.get("split_unit") != "sequence_or_scene":
        errors.append("data split must remain frozen at sequence_or_scene level")
    if split.get("prevent_adjacent_frame_leakage") is not True:
        errors.append("adjacent-frame leakage prevention must be enabled")

    references = experiment.get("references")
    if not isinstance(references, dict):
        errors.append("experiment references must be a mapping")
    else:
        for key in ("split_manifest", "model_config", "checkpoint_index", "report"):
            value = references.get(key)
            if not value or not (ROOT / str(value)).is_file():
                errors.append(f"experiment reference does not exist: {key}")

    randomness = experiment.get("randomness")
    if not isinstance(randomness, dict) or randomness.get("seed") != 20260716:
        errors.append("experiment seed must remain 20260716")
    thresholds = experiment.get("acceptance_thresholds")
    if not isinstance(thresholds, dict):
        errors.append("acceptance_thresholds must be a mapping")
    elif (
        thresholds.get("status") != "pending_product_and_target_hardware_confirmation"
        or thresholds.get("values") is not None
    ):
        errors.append("unapproved numeric acceptance thresholds must remain unset")


def _validate_model_and_metrics(
    experiment: dict[str, object],
    model_config: dict[str, object],
    errors: list[str],
) -> None:
    """交叉核对模型身份、输入尺寸、决策规则和三项主要指标。"""

    frozen_model = experiment.get("model")
    configured_model = model_config.get("model")
    if not isinstance(frozen_model, dict) or not isinstance(configured_model, dict):
        errors.append("model sections must be mappings")
        return
    for key in ("id", "revision", "weights_sha256"):
        if frozen_model.get(key) != configured_model.get(key):
            errors.append(f"frozen model {key} does not match task-5 model configuration")
    if frozen_model.get("local_training") is not False or frozen_model.get("parameters_frozen") is not True:
        errors.append("V1 model must remain inference-only with frozen parameters")

    input_config = experiment.get("input")
    if not isinstance(input_config, dict):
        errors.append("input configuration must be a mapping")
    elif (input_config.get("model_height"), input_config.get("model_width")) != (512, 512):
        errors.append("V1 model input must remain 512x512")

    inference = experiment.get("inference")
    if not isinstance(inference, dict):
        errors.append("inference configuration must be a mapping")
    elif inference.get("decision_rule") != "semantic_argmax" or inference.get("confidence_threshold") is not None:
        errors.append("V1 inference must use argmax without an invented confidence threshold")

    metrics = experiment.get("metrics")
    primary = metrics.get("primary", []) if isinstance(metrics, dict) else []
    metric_ids = {item.get("id") for item in primary if isinstance(item, dict)}
    if metric_ids != REQUIRED_PRIMARY_METRICS:
        errors.append(f"primary metrics must be exactly {sorted(REQUIRED_PRIMARY_METRICS)}")
    boundary = next((item for item in primary if item.get("id") == "boundary_fscore"), {})
    if boundary.get("tolerance_pixels") != 3 or boundary.get("boundary_connectivity") != 4:
        errors.append("boundary F-score must use 3-pixel tolerance and 4-connectivity")


def _validate_split(split: dict[str, object], check_local_files: bool, errors: list[str]) -> None:
    """检查样例数量、集合互斥、序列隔离及可选的本地文件完整性。"""

    partitions = split.get("partitions")
    if not isinstance(partitions, dict):
        errors.append("split partitions must be a mapping")
        return
    train = partitions.get("train", {})
    development = partitions.get("development_smoke", {})
    holdout = partitions.get("holdout_evaluation", {})
    if train.get("sample_count") != 0 or train.get("samples") != []:
        errors.append("V1 local training partition must remain empty")
    if development.get("sample_count") != 6 or holdout.get("sample_count") != 34:
        errors.append("V1 split must contain 6 development and 34 holdout samples")

    development_sets = development.get("datasets", [])
    holdout_sets = holdout.get("datasets", [])
    dev_laf = next((item for item in development_sets if item.get("id") == "lost-and-found"), {})
    test_laf = next((item for item in holdout_sets if item.get("id") == "lost-and-found"), {})
    road_obstacle = next((item for item in holdout_sets if item.get("id") == "road-obstacle-21"), {})
    dev_ids = set(dev_laf.get("sample_ids", []))
    test_ids = set(test_laf.get("sample_ids", []))
    if len(dev_ids) != 6 or len(test_ids) != 4 or dev_ids & test_ids:
        errors.append("Lost and Found sample ids must be unique and disjoint across partitions")
    if set(dev_laf.get("sequence_ids", [])) & set(test_laf.get("sequence_ids", [])):
        errors.append("Lost and Found development and holdout sequences must not overlap")
    scene_ids = road_obstacle.get("scene_ids", [])
    if road_obstacle.get("sample_count") != 30 or len(scene_ids) != 30 or len(set(scene_ids)) != 30:
        errors.append("RoadObstacle21 holdout must contain 30 unique public validation scenes")

    records = _task4_records()
    missing_ids = (dev_ids | test_ids) - records.keys()
    if missing_ids:
        errors.append(f"frozen Lost and Found samples missing from task-4 index: {sorted(missing_ids)}")
    if not check_local_files:
        return

    # 完整模式验证每个冻结样例的原图和语义掩码均真实存在。
    for sample_id in sorted(dev_ids | test_ids):
        record = records.get(sample_id)
        if not record:
            continue
        image_path = ROOT / str(record["image_path"])
        mask_path = ROOT / str(record["annotation_paths"]["semantic_mask"])
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"missing local Lost and Found files: {sample_id}")
    road_root = ROOT / "data_raw/segment-me-if-you-can/road-obstacle-21/extracted/dataset_ObstacleTrack"
    for scene_id in scene_ids:
        image_path = road_root / "images" / f"{scene_id}.webp"
        mask_path = road_root / "labels_masks" / f"{scene_id}_labels_semantic.png"
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"missing local RoadObstacle21 files: {scene_id}")


def _validate_local_processor(experiment: dict[str, object], errors: list[str]) -> None:
    """确认下载权重自带处理器确实使用冻结的 512×512 输入。"""

    model = experiment.get("model")
    if not isinstance(model, dict):
        errors.append("frozen model metadata must be a mapping")
        return
    model_directory = ROOT / "checkpoints/segformer-b0-cityscapes"
    processor_path = model_directory / "preprocessor_config.json"
    if not processor_path.is_file():
        errors.append("missing local SegFormer preprocessor_config.json")
        return
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    if processor.get("size") != 512:
        errors.append("local SegFormer processor size does not match frozen 512x512 input")
    if processor.get("do_resize") is not True or processor.get("do_normalize") is not True:
        errors.append("local SegFormer processor must keep resize and normalization enabled")
    if processor.get("image_mean") != [0.485, 0.456, 0.406]:
        errors.append("local SegFormer processor mean does not match the frozen experiment")
    if processor.get("image_std") != [0.229, 0.224, 0.225]:
        errors.append("local SegFormer processor std does not match the frozen experiment")
    if str(model.get("weights_sha256", "")) == "":
        errors.append("frozen model weight SHA256 must not be empty")


def validate(metadata_only: bool = False) -> list[str]:
    """运行实验方案校验；元数据模式不要求本地模型和原始样例存在。"""

    errors: list[str] = []
    try:
        experiment = _load_yaml(EXPERIMENT_PATH)
        split = _load_yaml(SPLIT_PATH)
        model_config = _load_yaml(MODEL_CONFIG_PATH)
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        return [f"cannot read experiment plan: {exc}"]
    if experiment.get("schema_version") != 1 or split.get("schema_version") != 1:
        errors.append("experiment and split schema_version must be 1")
    _validate_frozen_metadata(experiment, split, errors)
    _validate_model_and_metrics(experiment, model_config, errors)
    _validate_split(split, check_local_files=not metadata_only, errors=errors)
    if not metadata_only:
        _validate_local_processor(experiment, errors)
    return errors


def main() -> int:
    """解析校验范围，打印冻结方案验收结论并返回标准退出码。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    errors = validate(metadata_only=args.metadata_only)
    if errors:
        print("Experiment-plan validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "metadata" if args.metadata_only else "metadata, local model, and 40 local samples"
    print(f"Experiment-plan V1 validation passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
