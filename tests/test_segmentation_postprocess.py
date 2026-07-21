"""可行驶区域边界提取和叠加渲染的单元测试。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from driver_vision_risk.inference.drivable_area import (
    anomaly_eligible_region,
    categorical_boundary_band,
    extract_anomaly_regions,
    mask_anomaly_heatmap,
    render_anomaly_heatmap,
    render_overlay,
)
from driver_vision_risk.models.segformer import (
    DrivableAreaPrediction,
    binary_dilate,
    binary_erode,
    binary_inner_boundary,
)


def test_binary_inner_boundary_is_inside_mask() -> None:
    """内边界必须完全位于掩码内部，且不包含中心像素。"""

    mask = np.zeros((7, 7), dtype=np.bool_)
    mask[1:6, 1:6] = True
    boundary = binary_inner_boundary(mask, width=1)

    assert boundary.dtype == np.bool_
    assert np.all(boundary <= mask)
    assert int(boundary.sum()) == 16
    assert not boundary[3, 3]


def test_binary_erode_removes_requested_number_of_border_pixels() -> None:
    """道路掩码腐蚀应按四邻域移除边缘，并保留足够大的内部区域。"""

    mask = np.ones((7, 7), dtype=np.bool_)
    eroded = binary_erode(mask, width=2)

    assert int(eroded.sum()) == 9
    assert np.all(eroded[2:5, 2:5])
    assert not eroded[1, 3]


def test_binary_dilate_expands_diagonally_without_gaps() -> None:
    """八邻域膨胀应生成完整方形边界带，并覆盖对角像素。"""

    mask = np.zeros((5, 5), dtype=np.bool_)
    mask[2, 2] = True
    dilated = binary_dilate(mask, width=1)

    assert int(dilated.sum()) == 9
    assert dilated[1, 1]
    assert dilated[3, 3]


def test_render_overlay_marks_road_and_boundary() -> None:
    """叠加函数应保持背景不变，并用边界颜色覆盖道路边缘。"""

    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    boundary = binary_inner_boundary(mask)
    prediction = DrivableAreaPrediction(
        mask=mask,
        boundary=boundary,
        confidence=np.full((4, 4), 0.75, dtype=np.float32),
        anomaly=np.zeros((4, 4), dtype=np.float32),
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


def test_render_anomaly_heatmap_preserves_resolution() -> None:
    """异常热力图应保持分数图尺寸，并记录实际可视化范围。"""

    anomaly = np.arange(16, dtype=np.float32).reshape(4, 4)
    heatmap, scale = render_anomaly_heatmap(anomaly, 0.0, 100.0)

    assert heatmap.size == (4, 4)
    assert heatmap.mode == "RGB"
    assert scale["score_low"] == 0.0
    assert scale["score_high"] == 15.0


def test_mask_anomaly_heatmap_blacks_out_ineligible_pixels() -> None:
    """风险热力图应把道路 ROI 和类别边界带之外的区域显示为黑色。"""

    anomaly = np.arange(25, dtype=np.float32).reshape(5, 5)
    raw_heatmap, _ = render_anomaly_heatmap(anomaly, 0.0, 100.0)
    eligible = np.zeros((5, 5), dtype=np.bool_)
    eligible[1:4, 1:4] = True
    filtered = np.asarray(mask_anomaly_heatmap(raw_heatmap, eligible))

    assert np.all(filtered[0, :] == 0)
    assert np.all(filtered[:, 0] == 0)
    assert int(filtered[1:4, 1:4].sum()) > 0


def test_extract_anomaly_regions_filters_by_road_threshold_and_area() -> None:
    """连通域接口应忽略路外像素、小噪点，并输出明确的外接框和均值。"""

    mask = np.ones((8, 8), dtype=np.bool_)
    mask[:, 0] = False
    anomaly = np.zeros((8, 8), dtype=np.float32)
    anomaly[2:5, 2:5] = 2.0
    anomaly[6, 6] = 3.0
    anomaly[1:4, 0] = 5.0
    prediction = DrivableAreaPrediction(
        mask=mask,
        boundary=binary_inner_boundary(mask),
        confidence=np.ones((8, 8), dtype=np.float32),
        anomaly=anomaly,
        class_map=np.zeros((8, 8), dtype=np.uint8),
        latency_ms=1.0,
        device="cpu",
    )

    regions = extract_anomaly_regions(
        prediction,
        threshold=1.0,
        minimum_area_pixels=4,
        connectivity=8,
    )

    assert regions == [
        {
            "bbox_xyxy": [2, 2, 5, 5],
            "area_pixels": 9,
            "mean_anomaly_score": 2.0,
        }
    ]


def test_extract_anomaly_regions_ignores_eroded_road_edge() -> None:
    """腐蚀后的道路 ROI 应过滤路沿候选，同时保留道路内部异常。"""

    mask = np.ones((12, 12), dtype=np.bool_)
    anomaly = np.zeros((12, 12), dtype=np.float32)
    anomaly[0:2, 4:8] = 4.0
    anomaly[5:8, 5:8] = 2.0
    prediction = DrivableAreaPrediction(
        mask=mask,
        boundary=binary_inner_boundary(mask),
        confidence=np.ones((12, 12), dtype=np.float32),
        anomaly=anomaly,
        class_map=np.zeros((12, 12), dtype=np.uint8),
        latency_ms=1.0,
        device="cpu",
    )

    regions = extract_anomaly_regions(
        prediction,
        threshold=1.0,
        minimum_area_pixels=4,
        connectivity=8,
        road_mask_erosion_pixels=2,
    )

    assert regions == [
        {
            "bbox_xyxy": [5, 5, 8, 8],
            "area_pixels": 9,
            "mean_anomaly_score": 2.0,
        }
    ]


def test_class_boundary_band_suppresses_natural_boundary_energy() -> None:
    """类别交界带的高能量应被排除，道路类别内部异常仍应进入接口。"""

    mask = np.ones((16, 16), dtype=np.bool_)
    class_map = np.zeros((16, 16), dtype=np.uint8)
    class_map[:, 8:] = 1
    anomaly = np.zeros((16, 16), dtype=np.float32)
    anomaly[4:12, 6:10] = 4.0
    anomaly[2:4, 2:4] = 2.0
    prediction = DrivableAreaPrediction(
        mask=mask,
        boundary=binary_inner_boundary(mask),
        confidence=np.ones((16, 16), dtype=np.float32),
        anomaly=anomaly,
        class_map=class_map,
        latency_ms=1.0,
        device="cpu",
    )

    boundary_band = categorical_boundary_band(class_map, dilation_pixels=2)
    eligible = anomaly_eligible_region(
        prediction,
        road_mask_erosion_pixels=0,
        class_boundary_suppression_pixels=2,
    )
    regions = extract_anomaly_regions(
        prediction,
        threshold=1.0,
        minimum_area_pixels=4,
        connectivity=8,
        class_boundary_suppression_pixels=2,
    )

    assert boundary_band[:, 5:11].all()
    assert not eligible[:, 5:11].any()
    assert regions == [
        {
            "bbox_xyxy": [2, 2, 4, 4],
            "area_pixels": 4,
            "mean_anomaly_score": 2.0,
        }
    ]
