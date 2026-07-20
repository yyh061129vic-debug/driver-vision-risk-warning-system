"""支持通过 ``python -m driver_vision_risk`` 调用项目命令行。"""

from driver_vision_risk.cli import main


if __name__ == "__main__":
    # 把 CLI 返回码交给操作系统，便于脚本和 CI 判断执行是否成功。
    raise SystemExit(main())
