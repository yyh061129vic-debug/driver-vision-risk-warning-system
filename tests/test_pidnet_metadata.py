"""PIDNet-S 元数据与最小推理链路的回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pidnet_metadata() -> None:
    """PIDNet-S 的配置、许可和权重索引应保持一致。"""

    result = subprocess.run(
        [sys.executable, "scripts/validate_pidnet_demo.py", "--metadata-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
