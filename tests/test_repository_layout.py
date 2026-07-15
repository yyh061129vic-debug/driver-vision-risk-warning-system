from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_repository_layout() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_layout.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_yaml_files_are_valid() -> None:
    yaml_files = sorted(ROOT.glob("**/*.yaml"))
    assert yaml_files

    for path in yaml_files:
        with path.open(encoding="utf-8") as stream:
            assert yaml.safe_load(stream) is not None, path
