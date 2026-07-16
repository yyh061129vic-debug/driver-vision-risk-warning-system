"""任务 5 分割 Demo 的独立脚本入口。"""

from __future__ import annotations

import sys

from driver_vision_risk.cli import main


if __name__ == "__main__":
    # 复用正式 CLI，仅自动补上 ``segment`` 子命令，避免维护两套参数解析逻辑。
    raise SystemExit(main(["segment", *sys.argv[1:]]))
