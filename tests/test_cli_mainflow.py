"""主流程 CLI 的轻量回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_segment_help_lists_model_switch() -> None:
    """主流程帮助应公开内置模型切换参数。"""

    result = subprocess.run(
        [sys.executable, "-m", "driver_vision_risk", "segment", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "pidnet" in result.stdout
