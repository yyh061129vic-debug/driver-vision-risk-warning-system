"""林展毅：道路区域与道路标线检测合并接口。

另一组 YOLO 障碍物模块可以对同一帧并行推理，然后将检测框绘制到
本模块返回的 visualization 上。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch

from 林展毅_road_lane_detector import DetectorConfig, RoadLaneDetector


BASE_DIR = Path(__file__).resolve().parent
_detector: RoadLaneDetector | None = None
_init_lock = Lock()
_process_lock = Lock()


def _find_weight(pattern: str) -> Path:
    matches = sorted(BASE_DIR.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one weight matching {pattern!r} in {BASE_DIR}, "
            f"found {len(matches)}"
        )
    return matches[0]


def get_detector() -> RoadLaneDetector:
    """Create the detector once and reuse it for subsequent video frames."""
    global _detector
    if _detector is None:
        with _init_lock:
            if _detector is None:
                _detector = RoadLaneDetector(
                    DetectorConfig(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        road_model_path=str(
                            _find_weight("*_road_segformer_b2_bdd_best.pt")
                        ),
                        lane_model_path=str(
                            _find_weight("*_lane_segformer_b0_bdd_best.pt")
                        ),
                        road_probability_threshold=0.20,
                        lane_road_probability_threshold=0.05,
                        inference_height=384,
                        inference_width=672,
                        fixed_hood_crop=False,
                        enable_surface_markings=False,
                        observed_markings_only=False,
                        enable_low_light_enhancement=False,
                        enable_night_single_line_recovery=False,
                        road_inference_stride=1,
                        lane_inference_stride=1,
                    )
                )
    return _detector


def process_frame(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return road mask, lane-marking mask, and BGR visualization."""
    with _process_lock:
        result = get_detector().process_frame(frame)
    return result.road_mask, result.lane_mask, result.overlay


def reset_temporal_state() -> None:
    """Reset history before an unrelated image or a new video sequence."""
    with _process_lock:
        get_detector().reset_temporal_state()
