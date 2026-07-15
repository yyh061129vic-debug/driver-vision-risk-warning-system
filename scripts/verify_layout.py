"""Verify the repository scaffold without third-party dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "src/driver_vision_risk/data",
    "src/driver_vision_risk/models",
    "src/driver_vision_risk/training",
    "src/driver_vision_risk/inference",
    "src/driver_vision_risk/evaluation",
    "src/driver_vision_risk/interfaces",
    "src/driver_vision_risk/simulation",
    "src/driver_vision_risk/risk",
    "configs/data",
    "data/indexes",
    "data_raw",
    "data_processed",
    "metadata",
    "checkpoints",
    "outputs",
    "scripts",
    "tests",
    "docs",
)

REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "configs/paths.yaml",
    "configs/system.yaml",
    "configs/data/datasets.yaml",
    "data/indexes/manifest.schema.json",
    "metadata/dataset_registry.yaml",
    "checkpoints/index.yaml",
    "data_raw/.gitignore",
    "data_processed/.gitignore",
    "checkpoints/.gitignore",
    "outputs/.gitignore",
)

FORBIDDEN_TRACKABLE_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".engine",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".weights",
}


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED_DIRECTORIES:
        if not (ROOT / relative).is_dir():
            errors.append(f"missing directory: {relative}")

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing file: {relative}")

    schema_path = ROOT / "data/indexes/manifest.schema.json"
    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid data index schema: {exc}")
        else:
            if schema.get("type") != "object":
                errors.append("data index schema must describe an object")

    for parent in (ROOT / "data_raw", ROOT / "data_processed", ROOT / "checkpoints", ROOT / "outputs"):
        if not parent.is_dir():
            continue
        for path in parent.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_TRACKABLE_SUFFIXES:
                errors.append(f"large artifact found in Git scaffold: {path.relative_to(ROOT)}")

    if errors:
        print("Repository layout verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Repository layout verification passed.")
    print(f"Checked {len(REQUIRED_DIRECTORIES)} directories and {len(REQUIRED_FILES)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
