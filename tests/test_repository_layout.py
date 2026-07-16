"""仓库结构、元数据登记和任务验收脚本的回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_repository_layout() -> None:
    """目录验收脚本应在当前仓库布局下成功退出。"""

    result = subprocess.run(
        [sys.executable, "scripts/verify_layout.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_yaml_files_are_valid() -> None:
    """项目内所有 YAML 文件都应能被安全解析且内容非空。"""

    yaml_files = sorted(ROOT.glob("**/*.yaml"))
    assert yaml_files

    for path in yaml_files:
        with path.open(encoding="utf-8") as stream:
            assert yaml.safe_load(stream) is not None, path


def test_dataset_registry() -> None:
    """候选数据集登记与本地启用配置必须保持一致。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset_registry.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_environment_baseline() -> None:
    """环境快照必须完整、状态合法且不包含敏感机器信息。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_environment_baseline.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_task4_sample_metadata() -> None:
    """任务 4 配置和样例索引应通过轻量元数据校验。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_task4_visualizations.py", "--metadata-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_task5_segmentation_metadata() -> None:
    """任务 5 模型配置、许可和权重索引应保持一致。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_segmentation_demo.py", "--metadata-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_task6_experiment_plan_metadata() -> None:
    """任务 6 的数据、模型、输入尺寸和指标必须保持冻结且互相一致。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_experiment_plan.py", "--metadata-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
