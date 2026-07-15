"""Minimal project entry point for repository and environment inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driver_vision_risk import __version__


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _layout() -> dict[str, str]:
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
    parser = argparse.ArgumentParser(prog="driver-vision-risk")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--show-layout",
        action="store_true",
        help="print the resolved repository directories as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.show_layout:
        print(json.dumps(_layout(), ensure_ascii=False, indent=2))
        return 0

    build_parser().print_help()
    return 0
