"""项目命令行入口：查看目录布局并运行可行驶区域分割 Demo。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driver_vision_risk import __version__


def _project_root() -> Path:
    """根据已安装源码位置返回仓库根目录，避免硬编码本机绝对路径。"""

    return Path(__file__).resolve().parents[2]


def _layout() -> dict[str, str]:
    """生成核心目录的可序列化映射，供环境排查和外部脚本使用。"""

    root = _project_root()
    return {
        "project_root": str(root),
        "source": str(root / "src/driver_vision_risk"),
        "configs": str(root / "configs"),
        "data_raw": str(root / "data_raw"),
        "data_processed": str(root / "data_processed"),
        "data_indexes": str(root / "data/indexes"),
        "checkpoints": str(root / "checkpoints"),
        "outputs": str(root / "outputs"),
    }


def build_parser() -> argparse.ArgumentParser:
    """构建顶层参数解析器以及 ``segment`` 子命令。"""

    parser = argparse.ArgumentParser(prog="driver-vision-risk")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--show-layout",
        action="store_true",
        help="print the resolved repository directories as JSON",
    )
    subparsers = parser.add_subparsers(dest="command")
    segment = subparsers.add_parser(
        "segment",
        help="run the configured drivable-area segmentation demo",
    )
    segment.add_argument("--input", type=Path, required=True, help="input image or video path")
    segment.add_argument("--output", type=Path, required=True, help="output run directory")
    segment.add_argument(
        "--config",
        type=Path,
        default=_project_root() / "configs/models/segformer_cityscapes.yaml",
        help="segmentation model YAML configuration",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并分派命令，返回适合终端和自动化工具使用的退出码。"""

    args = build_parser().parse_args(argv)
    if args.show_layout:
        print(json.dumps(_layout(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "segment":
        # 延迟导入推理依赖，使目录查看和 ``--help`` 无需加载 PyTorch。
        from driver_vision_risk.inference.drivable_area import run_demo

        result = run_demo(args.input, args.output, args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    build_parser().print_help()
    return 0
