"""在不加载第三方库的情况下校验仓库目录骨架和大文件隔离。"""

from __future__ import annotations

import json
import subprocess
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
    "configs/experiments",
    "configs/models",
    "data/indexes",
    "data_raw",
    "data_processed",
    "metadata",
    "metadata/environment",
    "checkpoints",
    "outputs",
    "scripts",
    "tests",
    "team_submissions",
    "team_submissions/example",
    "docs",
)

REQUIRED_FILES = (
    "README.md",
    "team_submissions/README.md",
    "team_submissions/example/README.md",
    "pyproject.toml",
    "configs/paths.yaml",
    "configs/system.yaml",
    "configs/data/datasets.yaml",
    "configs/data/task4_samples.yaml",
    "configs/experiments/drivable_area_v1.yaml",
    "configs/models/segformer_cityscapes.yaml",
    "configs/models/segformer_cityscapes_cpu.yaml",
    "data/indexes/manifest.schema.json",
    "data/indexes/task4_samples.jsonl",
    "data/indexes/drivable_area_v1_split.yaml",
    "metadata/dataset_registry.yaml",
    "metadata/licenses/README.md",
    "metadata/licenses/lost-and-found-2026-07-16.md",
    "metadata/licenses/road-obstacle-21-2026-07-16.md",
    "metadata/licenses/segformer-b0-cityscapes-2026-07-16.md",
    "metadata/environment/README.md",
    "metadata/environment/baseline-2026-07-16.yaml",
    "docs/dataset-survey.md",
    "docs/environment-baseline.md",
    "docs/sample-visualization.md",
    "docs/segmentation-demo.md",
    "docs/experiment-plan-v1.md",
    "scripts/download_segmentation_model.py",
    "scripts/download_task4_samples.py",
    "scripts/validate_dataset_registry.py",
    "scripts/validate_environment_baseline.py",
    "scripts/visualize_dataset_samples.py",
    "scripts/validate_task4_visualizations.py",
    "scripts/run_segmentation_demo.py",
    "scripts/validate_segmentation_demo.py",
    "scripts/validate_experiment_plan.py",
    "src/driver_vision_risk/inference/drivable_area.py",
    "src/driver_vision_risk/models/segformer.py",
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
    """检查必需目录、文件、索引模式以及 Git 中的禁止产物。"""

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

    # 只扫描 Git 已跟踪文件，忽略本地数据、权重和运行输出本身。
    tracked = subprocess.run(
        ["git", "ls-files", "--", "data_raw", "data_processed", "checkpoints", "outputs"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode == 0:
        for relative in tracked.stdout.splitlines():
            path = ROOT / relative
            if path.suffix.lower() in FORBIDDEN_TRACKABLE_SUFFIXES:
                errors.append(f"large artifact tracked by Git: {relative}")

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
