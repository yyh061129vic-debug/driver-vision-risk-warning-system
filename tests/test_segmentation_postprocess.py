"""可行驶区域边界提取和叠加渲染的单元测试。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from driver_vision_risk.inference.drivable_area import render_overlay
from driver_vision_risk.models.segformer import DrivableAreaPrediction, binary_inner_boundary


def test_binary_inner_boundary_is_inside_mask() -> None:
    """内边界必须完全位于掩码内部，且不包含中心像素。"""

    mask = np.zeros((7, 7), dtype=np.bool_)
    mask[1:6, 1:6] = True
    boundary = binary_inner_boundary(mask, width=1)

    assert boundary.dtype == np.bool_
    assert np.all(boundary <= mask)
    assert int(boundary.sum()) == 16
    assert not boundary[3, 3]


def test_render_overlay_marks_road_and_boundary() -> None:
    """叠加函数应保持背景不变，并用边界颜色覆盖道路边缘。"""

    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    boundary = binary_inner_boundary(mask)
    prediction = DrivableAreaPrediction(
        mask=mask,
        boundary=boundary,
        confidence=np.full((4, 4), 0.75, dtype=np.float32),
        class_map=np.zeros((4, 4), dtype=np.uint8),
        latency_ms=1.0,
        device="cpu",
    )
    source = Image.new("RGB", (4, 4), (100, 100, 100))
    overlay = np.asarray(
        render_overlay(source, prediction, (0, 200, 0), (255, 255, 255), 0.5)
    )

    assert tuple(overlay[0, 0]) == (100, 100, 100)
    assert tuple(overlay[1, 1]) == (255, 255, 255)
