"""校验 PIDNet-S 模型配置、固定权重和 Demo 产物。"""

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
CONFIG_PATH = ROOT / "configs/models/pidnet_s_cityscapes.yaml"
CHECKPOINT_INDEX = ROOT / "checkpoints/index.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/task5_pidnet_demo/local"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(config: dict[str, object], errors: list[str]) -> None:
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
        "weights_sha256",
        "external_data_file",
        "external_data_sha256",
        "metadata_file",
        "metadata_sha256",
        "labels_file",
        "labels_sha256",
        "asset_url",
        "license_snapshot",
    }
    missing = required - model.keys()
    if missing:
        errors.append(f"PIDNet configuration missing fields: {sorted(missing)}")
    license_snapshot = model.get("license_snapshot")
    if not license_snapshot or not (ROOT / str(license_snapshot)).is_file():
        errors.append("PIDNet license snapshot must reference an existing file")

    segmentation = config.get("segmentation")
    if not isinstance(segmentation, dict):
        errors.append("segmentation configuration must be a mapping")
    else:
        if segmentation.get("road_class_ids") != [0]:
            errors.append("PIDNet road class id must be 0")
        if segmentation.get("road_class_names") != ["road"]:
            errors.append("PIDNet road class name must be road")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("framework") != "onnxruntime":
        errors.append("PIDNet runtime framework must be onnxruntime")

    index = yaml.safe_load(CHECKPOINT_INDEX.read_text(encoding="utf-8"))
    entries = index.get("models", []) if isinstance(index, dict) else []
    matches = [entry for entry in entries if entry.get("id") == model.get("id")]
    if len(matches) != 1:
        errors.append("checkpoint index must contain exactly one configured PIDNet model")
        return
    entry = matches[0]
    if entry.get("source", {}).get("revision") != model.get("revision"):
        errors.append("checkpoint index revision does not match PIDNet configuration")
    if entry.get("weights", {}).get("sha256") != model.get("weights_sha256"):
        errors.append("checkpoint index SHA256 does not match PIDNet configuration")


def _validate_checkpoint(config: dict[str, object], errors: list[str]) -> None:
    model = config["model"]
    directory = ROOT / model["local_directory"]
    files = (
        (model["weights_file"], model["weights_sha256"], int(model["weights_size_bytes"])),
        (model["external_data_file"], model["external_data_sha256"], int(model["external_data_size_bytes"])),
        (model["metadata_file"], model["metadata_sha256"], None),
        (model["labels_file"], model["labels_sha256"], None),
    )
    for filename, expected_sha, expected_size in files:
        path = directory / str(filename)
        if not path.is_file():
            errors.append(f"missing PIDNet checkpoint file: {path.relative_to(ROOT)}")
            continue
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(f"PIDNet size mismatch for {filename}")
        if _sha256(path) != expected_sha:
            errors.append(f"PIDNet SHA256 mismatch for {filename}")


def _validate_image_output(output_directory: Path, result: dict[str, object], errors: list[str]) -> None:
    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("result outputs must be a mapping")
        return
    images: dict[str, Image.Image] = {}
    for key in ("mask", "boundary", "confidence", "overlay"):
        filename = outputs.get(key)
        path = output_directory / str(filename)
        if not filename or not path.is_file():
            errors.append(f"missing PIDNet {key} output")
            continue
        expected_hash = outputs.get(f"{key}_sha256")
        if expected_hash != _sha256(path):
            errors.append(f"PIDNet {key} SHA256 mismatch")
        images[key] = Image.open(path).copy()
    if len(images) != 4:
        return
    dimensions = {image.size for image in images.values()}
    if len(dimensions) != 1:
        errors.append("PIDNet image outputs must have identical dimensions")
    mask = np.asarray(images["mask"])
    boundary = np.asarray(images["boundary"])
    confidence = np.asarray(images["confidence"])
    if not set(np.unique(mask)).issubset({0, 255}) or mask.max() == 0:
        errors.append("PIDNet mask must be a non-empty binary image")
    if not set(np.unique(boundary)).issubset({0, 255}) or boundary.max() == 0:
        errors.append("PIDNet boundary must be a non-empty binary image")
    if confidence.max() <= confidence.min():
        errors.append("PIDNet confidence image must contain a non-constant score field")
    input_path = ROOT / result["input"]["path"]
    if input_path.is_file():
        source = Image.open(input_path).convert("RGB")
        overlay = images["overlay"].convert("RGB")
        if source.size == overlay.size and ImageChops.difference(source, overlay).getbbox() is None:
            errors.append("PIDNet overlay must differ from the source image")


def validate(output_directory: Path, metadata_only: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read PIDNet configuration: {exc}"]
    if not isinstance(config, dict):
        return ["PIDNet configuration root must be a mapping"]
    if config.get("schema_version") != 1:
        errors.append("PIDNet configuration schema_version must be 1")
    _validate_metadata(config, errors)
    if metadata_only:
        return errors

    _validate_checkpoint(config, errors)
    result_path = output_directory / "result.json"
    if not result_path.is_file():
        errors.append("missing PIDNet result.json")
        return errors
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = result.get("model", {})
    if model.get("revision") != config["model"]["revision"]:
        errors.append("PIDNet result revision does not match configured checkpoint")
    if model.get("weights_sha256") != config["model"]["weights_sha256"]:
        errors.append("PIDNet result weight SHA256 does not match configured checkpoint")
    runtime = result.get("runtime", {})
    if runtime.get("framework") != "onnxruntime":
        errors.append("PIDNet result must record onnxruntime framework")
    if float(runtime.get("latency_ms", 0.0)) <= 0:
        errors.append("PIDNet result must record positive end-to-end latency")
    coverage = float(runtime.get("road_coverage", -1.0))
    if not 0.0 < coverage < 1.0:
        errors.append("PIDNet road coverage must be between zero and one")
    if result.get("input", {}).get("type") == "image":
        _validate_image_output(output_directory, result, errors)
    else:
        errors.append("full PIDNet validation currently expects the accepted image demo")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    errors = validate(args.output.resolve(), metadata_only=args.metadata_only)
    if errors:
        print("PIDNet validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "configuration and checkpoint index" if args.metadata_only else "checkpoint and demo outputs"
    print(f"PIDNet validation passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
