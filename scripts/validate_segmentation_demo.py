"""校验任务 5 模型配置、固定权重和分割 Demo 产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/models/segformer_cityscapes.yaml"
CHECKPOINT_INDEX = ROOT / "checkpoints/index.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/task5_segmentation_demo/gpu"


def _sha256(path: Path) -> str:
    """分块计算模型或输出文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(config: dict[str, object], errors: list[str]) -> None:
    """校验模型字段、道路类别、许可快照及权重索引的一致性。"""

    model = config.get("model")
    if not isinstance(model, dict):
        errors.append("model configuration must be a mapping")
        return
    required = {
        "id",
        "architecture",
        "repository",
        "revision",
        "local_directory",
        "weights_file",
        "weights_size_bytes",
        "weights_sha256",
        "license_snapshot",
    }
    missing = required - model.keys()
    if missing:
        errors.append(f"model configuration missing fields: {sorted(missing)}")
    if len(str(model.get("revision", ""))) != 40:
        errors.append("model revision must be a full 40-character commit hash")
    license_snapshot = model.get("license_snapshot")
    if not license_snapshot or not (ROOT / str(license_snapshot)).is_file():
        errors.append("model license snapshot must reference an existing file")

    segmentation = config.get("segmentation")
    if not isinstance(segmentation, dict):
        errors.append("segmentation configuration must be a mapping")
    else:
        if segmentation.get("road_class_ids") != [0]:
            errors.append("task-5 baseline must use checkpoint road class id 0")
        if segmentation.get("road_class_names") != ["road"]:
            errors.append("task-5 baseline must use checkpoint road class name road")

    # 配置与模型索引交叉核对，避免 revision 或摘要只更新一处。
    index = yaml.safe_load(CHECKPOINT_INDEX.read_text(encoding="utf-8"))
    entries = index.get("models", []) if isinstance(index, dict) else []
    matches = [entry for entry in entries if entry.get("id") == model.get("id")]
    if len(matches) != 1:
        errors.append("checkpoint index must contain exactly one configured task-5 model")
        return
    entry = matches[0]
    if entry.get("source", {}).get("revision") != model.get("revision"):
        errors.append("checkpoint index revision does not match model configuration")
    if entry.get("weights", {}).get("sha256") != model.get("weights_sha256"):
        errors.append("checkpoint index SHA256 does not match model configuration")


def _validate_checkpoint(config: dict[str, object], errors: list[str]) -> None:
    """确认本地模型文件齐全，并核对权重大小和 SHA-256。"""

    model = config["model"]
    directory = ROOT / model["local_directory"]
    for filename in ("config.json", "preprocessor_config.json", model["weights_file"]):
        if not (directory / filename).is_file():
            errors.append(f"missing checkpoint file: {(directory / filename).relative_to(ROOT)}")
    weights = directory / model["weights_file"]
    if weights.is_file():
        if weights.stat().st_size != int(model["weights_size_bytes"]):
            errors.append("checkpoint weight size mismatch")
        if _sha256(weights) != model["weights_sha256"]:
            errors.append("checkpoint weight SHA256 mismatch")


def _validate_image_output(
    output_directory: Path,
    result: dict[str, object],
    errors: list[str],
) -> None:
    """检查单图模式四类输出的哈希、尺寸、像素内容及叠加有效性。"""

    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("result outputs must be a mapping")
        return
    images: dict[str, Image.Image] = {}
    # 每类产物都必须存在且与 result.json 中记录的哈希一致。
    for key in ("mask", "boundary", "confidence", "overlay"):
        filename = outputs.get(key)
        path = output_directory / str(filename)
        if not filename or not path.is_file():
            errors.append(f"missing task-5 {key} output")
            continue
        expected_hash = outputs.get(f"{key}_sha256")
        if expected_hash != _sha256(path):
            errors.append(f"task-5 {key} SHA256 mismatch")
        try:
            images[key] = Image.open(path).copy()
        except OSError as exc:
            errors.append(f"cannot open task-5 {key}: {exc}")
    if len(images) != 4:
        return
    dimensions = {image.size for image in images.values()}
    if len(dimensions) != 1:
        errors.append("task-5 image outputs must have identical dimensions")
    # 防止“生成了文件”但掩码为空、边界为空或置信度恒定的伪通过。
    mask = np.asarray(images["mask"])
    boundary = np.asarray(images["boundary"])
    confidence = np.asarray(images["confidence"])
    if not set(np.unique(mask)).issubset({0, 255}) or mask.max() == 0:
        errors.append("drivable mask must be a non-empty binary image")
    if not set(np.unique(boundary)).issubset({0, 255}) or boundary.max() == 0:
        errors.append("drivable boundary must be a non-empty binary image")
    if confidence.max() <= confidence.min():
        errors.append("road confidence image must contain a non-constant score field")

    # 原图仍在本地时，确认叠加图确实发生像素变化。
    input_path = ROOT / result["input"]["path"]
    if input_path.is_file():
        source = Image.open(input_path).convert("RGB")
        overlay = images["overlay"].convert("RGB")
        if source.size == overlay.size and ImageChops.difference(source, overlay).getbbox() is None:
            errors.append("overlay must differ from the source image")


def validate(output_directory: Path, metadata_only: bool = False) -> list[str]:
    """执行任务 5 分层校验；元数据模式不要求本地权重和运行产物。"""

    errors: list[str] = []
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read segmentation configuration: {exc}"]
    if not isinstance(config, dict):
        return ["segmentation configuration root must be a mapping"]
    if config.get("schema_version") != 1:
        errors.append("segmentation configuration schema_version must be 1")
    _validate_metadata(config, errors)
    if metadata_only:
        return errors

    # 完整验收必须同时验证权重、运行记录以及可视化像素内容。
    _validate_checkpoint(config, errors)
    result_path = output_directory / "result.json"
    if not result_path.is_file():
        errors.append("missing task-5 result.json")
        return errors
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = result.get("model", {})
    if model.get("revision") != config["model"]["revision"]:
        errors.append("demo result revision does not match configured checkpoint")
    if model.get("weights_sha256") != config["model"]["weights_sha256"]:
        errors.append("demo result weight SHA256 does not match configured checkpoint")
    runtime = result.get("runtime", {})
    configured_device = config.get("runtime", {}).get("device")
    if runtime.get("device") != configured_device:
        errors.append("demo result device does not match configured runtime device")
    if float(runtime.get("latency_ms", 0.0)) <= 0:
        errors.append("demo result must record positive end-to-end latency")
    coverage = float(runtime.get("road_coverage", -1.0))
    if not 0.0 < coverage < 1.0:
        errors.append("demo road coverage must be between zero and one")
    if result.get("input", {}).get("type") == "image":
        _validate_image_output(output_directory, result, errors)
    else:
        errors.append("full task-5 validation currently expects the accepted image demo")
    return errors


def main() -> int:
    """解析输出目录和校验范围，并打印任务 5 验收结论。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    errors = validate(args.output.resolve(), metadata_only=args.metadata_only)
    if errors:
        print("Task-5 segmentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "configuration and checkpoint index" if args.metadata_only else "checkpoint and demo outputs"
    print(f"Task-5 segmentation validation passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
