"""命令行入口 ``driver-vision-risk`` 的参数解析与输出测试。"""

from __future__ import annotations

import json

import pytest

from driver_vision_risk import __version__
from driver_vision_risk.cli import _layout, build_parser, main


def test_show_layout_outputs_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``--show-layout`` 应输出包含全部核心目录的合法 JSON 并成功退出。"""

    assert main(["--show-layout"]) == 0

    layout = json.loads(capsys.readouterr().out)
    assert layout == _layout()
    assert set(layout) == {
        "project_root",
        "source",
        "configs",
        "data_raw",
        "data_processed",
        "data_indexes",
        "checkpoints",
        "outputs",
    }


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """不带参数运行时应打印帮助信息且不报错退出。"""

    assert main([]) == 0
    assert "usage: driver-vision-risk" in capsys.readouterr().out


def test_version_flag_reports_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` 应输出与包元数据一致的版本号。"""

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_segment_requires_input_and_output() -> None:
    """``segment`` 子命令缺少必填参数时应以参数错误退出。"""

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["segment"])

    assert excinfo.value.code == 2
