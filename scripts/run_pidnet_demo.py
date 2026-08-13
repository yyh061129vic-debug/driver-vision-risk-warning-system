"""PIDNet-S 可行驶区域 Demo 的独立脚本入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from driver_vision_risk.cli import main


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/models/pidnet_s_cityscapes.yaml"


if __name__ == "__main__":
    raise SystemExit(main(["segment", "--config", str(DEFAULT_CONFIG), *sys.argv[1:]]))
