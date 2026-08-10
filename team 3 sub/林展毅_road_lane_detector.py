"""Road and longitudinal lane-boundary detection for fixed dash cameras."""

# 作者：林展毅
# 功能：道路区域分割、道路标线检测及逐帧可视化。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np


def resolve_road_class_id(
    road_class_id: int | None,
    road_model_path: str | None,
    checkpoint_metadata: dict | None = None,
) -> int:
    """Resolve the road class for a binary segmentation head.

    BDD checkpoints produced by the local trainer encode road as class 1,
    while the Cityscapes model uses class 0.  Explicit configuration wins,
    followed by checkpoint metadata, then the local-checkpoint heuristic.
    """
    metadata_id = (
        checkpoint_metadata.get("road_class_id")
        if checkpoint_metadata is not None
        else None
    )
    selected = road_class_id if road_class_id is not None else metadata_id
    if selected is None:
        selected = 1 if road_model_path and Path(road_model_path).exists() else 0
    if selected not in (0, 1):
        raise ValueError("road_class_id must be 0 or 1")
    return int(selected)


def road_mask_from_prediction(prediction: np.ndarray, road_class_id: int) -> np.ndarray:
    """Convert argmax class predictions to a binary road mask."""
    if road_class_id not in (0, 1):
        raise ValueError("road_class_id must be 0 or 1")
    return (prediction == road_class_id).astype(np.uint8) * 255


def road_mask_from_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Convert binary-road probabilities to a uint8 mask."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("road_probability_threshold must be between 0 and 1")
    return (probability >= threshold).astype(np.uint8) * 255


def load_road_checkpoint(checkpoint_path: Path, torch_module):
    """Load a locally produced road checkpoint including resume metadata."""
    return torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def extract_surface_markings(frame_bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Extract broad, near-horizontal paint inside the final road corridor.

    This deliberately does not attempt to fit longitudinal lane geometry: a
    horizontal morphology kernel keeps stop bars and zebra stripes while
    suppressing thin vertical lane lines, arrows, and out-of-road clutter.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image")
    if road_mask.ndim != 2 or road_mask.shape != frame_bgr.shape[:2]:
        raise ValueError("road_mask must match frame dimensions")
    height, width = road_mask.shape
    if height == 0 or width == 0 or not np.any(road_mask):
        return np.zeros_like(road_mask, dtype=np.uint8)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 175], dtype=np.uint8),
        np.array([180, 65, 255], dtype=np.uint8),
    )
    yellow = cv2.inRange(
        hsv,
        np.array([15, 70, 90], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )
    # Scale the horizontal support with resolution, retaining a few pixels on
    # small test frames while requiring meaningful continuity on HD frames.
    kernel_width = max(5, min(41, int(round(width / 20)) | 1))
    horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, max(3, min(7, height // 35 | 1)))
    )
    min_width = max(6, width // 30)
    min_y = int(round(height * 0.40))
    road_area = max(1, int(np.count_nonzero(road_mask)))
    road_columns = np.flatnonzero(np.any(road_mask > 0, axis=0))
    road_width = (
        max(1, int(road_columns[-1] - road_columns[0] + 1))
        if road_columns.size
        else width
    )

    def extract_components(color_mask: np.ndarray, yellow_mode: bool) -> np.ndarray:
        candidates = cv2.bitwise_and(color_mask, road_mask)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, horizontal)
        candidates = cv2.morphologyEx(
            candidates,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )
        binary = (candidates > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        eligible: list[int] = []
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if y + component_height <= min_y:
                continue
            if (
                component_width < min_width
                or component_width / max(1, component_height) < 2.5
                or area < max(20, component_width * 1.5)
            ):
                continue
            if not yellow_mode:
                # Bright achromatic road glare/hood surfaces often survive the
                # horizontal opening as one broad connected component.  Keep
                # thin stop bars (including nearly road-width bars), but reject
                # components that occupy a large fraction of the road corridor
                # in both area and vertical extent.
                area_ratio = area / road_area
                width_ratio = component_width / road_width
                height_ratio = component_height / max(1, height)
                broad_fill = (
                    area_ratio > 0.20
                    or (width_ratio >= 0.80 and height_ratio >= 0.15)
                    or (width_ratio >= 0.95 and component_height > max(10, height // 20))
                )
                if broad_fill:
                    continue
            eligible.append(label)

        if yellow_mode:
            # A lone yellow broad patch is commonly glare, a hood reflection,
            # or text. Keep yellow only when it forms a repeated set of
            # similarly sized horizontal bands, or is an unmistakably long
            # stop bar.
            accepted: set[int] = set()
            for label in eligible:
                _, _, component_width, component_height, _ = stats[label]
                if (
                    component_width >= max(min_width * 2, int(round(width * 0.65)))
                    and component_width / max(1, component_height) >= 4.0
                ):
                    accepted.add(label)
            for index, first_label in enumerate(eligible):
                first_x, first_y, first_width, first_height, _ = stats[first_label]
                first_cx = first_x + first_width / 2.0
                first_cy = first_y + first_height / 2.0
                for second_label in eligible[index + 1 :]:
                    second_x, second_y, second_width, second_height, _ = stats[
                        second_label
                    ]
                    second_cx = second_x + second_width / 2.0
                    second_cy = second_y + second_height / 2.0
                    y_gap = abs(first_cy - second_cy)
                    x_gap = abs(first_cx - second_cx)
                    width_similar = abs(first_width - second_width) <= max(
                        4, max(first_width, second_width) * 0.30
                    )
                    height_similar = abs(first_height - second_height) <= max(
                        3, max(first_height, second_height) * 0.60
                    )
                    if (
                        y_gap <= max(12, height * 0.20)
                        and x_gap <= max(12, width * 0.12)
                        and width_similar
                        and height_similar
                    ):
                        accepted.update((first_label, second_label))
            eligible = [label for label in eligible if label in accepted]

        output = np.zeros_like(road_mask, dtype=np.uint8)
        for label in eligible:
            output[labels == label] = 255
        return output

    output = cv2.bitwise_or(
        extract_components(white, yellow_mode=False),
        extract_components(yellow, yellow_mode=True),
    )
    return cv2.bitwise_and(output, road_mask)


def is_low_light_scene(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray | None = None,
) -> bool:
    """Classify scenes from the dark-pixel distribution, not mean brightness.

    Headlights can raise the global mean substantially while most of the road
    remains dark.  A low gray percentile and dark-pixel ratio capture that
    pattern; a bright-pixel ratio prevents ordinary daylight from being
    classified as night.  When available, the road corridor is preferred over
    sky/vehicle pixels outside it.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if road_mask is not None:
        if road_mask.ndim != 2 or road_mask.shape != gray.shape:
            raise ValueError("road_mask must match frame dimensions")
        region = road_mask > 0
        # Ignore an unusably sparse prediction and fall back to the frame.
        if np.count_nonzero(region) >= max(32, gray.size * 0.05):
            values = gray[region]
        else:
            values = gray.reshape(-1)
    else:
        values = gray.reshape(-1)
    if values.size == 0:
        return False

    low_percentile = float(np.percentile(values, 20))
    dark_ratio = float(np.mean(values <= 55))
    bright_ratio = float(np.mean(values >= 180))
    # Permit bright headlights only when a substantial dark background remains.
    return bool(
        low_percentile <= 50.0
        and dark_ratio >= 0.35
        and (bright_ratio <= 0.45 or dark_ratio >= 0.55)
    )


def enhance_low_light_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply a restrained luminance enhancement for learned lane inference.

    Only the LAB lightness channel is equalized, leaving chroma untouched so
    headlight colour does not become a lane cue.  The conservative CLAHE
    settings improve local contrast in dark pavement while avoiding the large
    global gain that can turn headlights into saturated blobs.
    """
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError("frame_bgr must be a BGR uint8 image with three channels")
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    equalized = clahe.apply(lightness)
    # Blend rather than replacing L: this keeps bright sources close to their
    # original intensity and limits night-time halo amplification.
    lab[:, :, 0] = cv2.addWeighted(lightness, 0.35, equalized, 0.65, 0.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_pretrained_cached_first(factory, model_name: str, **kwargs):
    """Load cached Hugging Face assets before attempting network access."""
    try:
        return factory.from_pretrained(
            model_name,
            **kwargs,
            local_files_only=True,
        )
    except OSError:
        return factory.from_pretrained(model_name, **kwargs)


def build_camera_roi(
    shape: tuple[int, int],
    hood_top_ratio: float,
) -> np.ndarray:
    """Return the valid scene area, excluding the fixed ego-vehicle hood."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    if not 0.5 <= hood_top_ratio <= 1.0:
        raise ValueError("hood_top_ratio must be between 0.5 and 1.0")

    roi = np.full((height, width), 255, dtype=np.uint8)
    roi[int(round(height * hood_top_ratio)) :] = 0
    return roi


def select_seeded_component(
    mask: np.ndarray,
    seed_y_ratio: float = 0.68,
    seed_width_ratio: float = 0.20,
) -> np.ndarray:
    """Keep the road component connected to the lower image centre."""
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")

    binary = (mask > 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary * 255

    height, width = binary.shape
    y = min(height - 1, int(round(height * seed_y_ratio)))
    half_width = max(1, int(round(width * seed_width_ratio / 2)))
    x_start = max(0, width // 2 - half_width)
    x_end = min(width, width // 2 + half_width)
    seed_labels = labels[max(0, y - 2) : min(height, y + 3), x_start:x_end]
    candidates = [value for value in np.unique(seed_labels) if value != 0]
    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8)

    selected_label = max(
        candidates,
        key=lambda value: np.count_nonzero(labels == value),
    )
    return (labels == selected_label).astype(np.uint8) * 255


def trace_road_corridor(mask: np.ndarray) -> np.ndarray:
    """Keep a geometrically continuous road corridor seeded near the camera."""
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")

    binary = mask > 0
    height, width = binary.shape
    traced = np.zeros_like(mask, dtype=np.uint8)
    if height == 0 or width == 0 or not np.any(binary):
        return traced

    padded = np.zeros((height, width + 2), dtype=bool)
    padded[:, 1:-1] = binary
    boundaries = padded[:, 1:] != padded[:, :-1]
    runs_by_row = []
    for row_boundaries in boundaries:
        edges = np.flatnonzero(row_boundaries)
        runs_by_row.append(
            list(zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()))
        )

    def row_runs(y: int) -> list[tuple[int, int]]:
        return runs_by_row[y]

    seed_top = min(height - 1, int(round(height * 0.55)))
    seed_bottom = min(height, max(seed_top + 1, int(round(height * 0.95))))
    support_half_width = max(1, int(round(width * 0.10)))
    support_left = max(0, width // 2 - support_half_width)
    support_right = min(width - 1, width // 2 + support_half_width)
    min_seed_rows = max(3, height // 60)

    def runs_are_consistent(
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> bool:
        first_left, first_right = first
        second_left, second_right = second
        first_width = first_right - first_left + 1
        second_width = second_right - second_left + 1
        return (
            first_left <= second_right
            and first_right >= second_left
            and abs(first_width - second_width)
            <= max(2, max(first_width, second_width) * 0.20)
        )

    forward_tracks = []
    for y, runs in enumerate(runs_by_row):
        previous = runs_by_row[y - 1] if y else []
        previous_lengths = forward_tracks[y - 1] if y else []
        forward_tracks.append([
            1 + max(
                (
                    previous_lengths[index]
                    for index, previous_run in enumerate(previous)
                    if runs_are_consistent(run, previous_run)
                ),
                default=0,
            )
            for run in runs
        ])

    backward_tracks = [[] for _ in range(height)]
    for y in range(height - 1, -1, -1):
        following = runs_by_row[y + 1] if y + 1 < height else []
        following_lengths = backward_tracks[y + 1] if y + 1 < height else []
        backward_tracks[y] = [
            1 + max(
                (
                    following_lengths[index]
                    for index, following_run in enumerate(following)
                    if runs_are_consistent(run, following_run)
                ),
                default=0,
            )
            for run in runs_by_row[y]
        ]

    seed_candidates = []
    min_seed_width = max(3, int(round(width * 0.08)))
    for y in range(seed_top, seed_bottom):
        for run_index, (left, right) in enumerate(row_runs(y)):
            run_width = right - left + 1
            forward = forward_tracks[y][run_index]
            backward = backward_tracks[y][run_index]
            track_length = forward + backward - 1
            if (
                run_width >= min_seed_width
                and track_length >= min_seed_rows
                and left <= support_right
                and right >= support_left
            ):
                centre_distance = abs((left + right) / 2.0 - width / 2.0)
                score = (
                    y
                    + min(run_width, width * 0.25) * 0.15
                    + track_length * 0.15
                    - centre_distance * 0.25
                )
                seed_candidates.append(
                    (score, y, -centre_distance, run_width, left, right)
                )
    if not seed_candidates:
        return traced

    _, seed_y, _, _, seed_left, seed_right = max(seed_candidates)
    traced[seed_y, seed_left : seed_right + 1] = 255
    max_centre_step = max(1.0, width * 0.025)
    max_half_width_step = max(1.0, width * 0.035)
    max_missing = max(5, height // 30)

    def follow(rows) -> None:
        centre = (seed_left + seed_right) / 2.0
        half_width = (seed_right - seed_left + 1) / 2.0
        observed_y = seed_y
        observed_left = seed_left
        observed_right = seed_right
        missing = 0
        for y in rows:
            candidates = []
            for left, right in row_runs(y):
                if left > centre + half_width or right < centre - half_width:
                    continue
                candidate_centre = (left + right) / 2.0
                candidate_half_width = (right - left + 1) / 2.0
                elapsed = missing + 1
                if (
                    abs(candidate_centre - centre) <= max_centre_step * elapsed
                    and abs(candidate_half_width - half_width)
                    <= max_half_width_step * elapsed
                ):
                    overlap = max(0, min(right, observed_right) - max(left, observed_left) + 1)
                    union = max(right, observed_right) - min(left, observed_left) + 1
                    iou = overlap / union
                    candidates.append(
                        (
                            iou,
                            -abs(candidate_centre - centre),
                            -abs(candidate_half_width - half_width),
                            (right - left + 1) * 0.01,
                            left,
                            right,
                            candidate_centre,
                            candidate_half_width,
                        )
                    )
            if not candidates:
                missing += 1
                if missing > max_missing:
                    break
                continue

            *_, left, right, centre, half_width = max(candidates)
            if missing:
                gap = abs(y - observed_y)
                for step in range(1, gap):
                    ratio = step / gap
                    fill_y = observed_y + (step if y > observed_y else -step)
                    fill_left = int(round(observed_left + (left - observed_left) * ratio))
                    fill_right = int(round(observed_right + (right - observed_right) * ratio))
                    traced[fill_y, fill_left : fill_right + 1] = 255
            traced[y, left : right + 1] = 255
            observed_y = y
            observed_left = left
            observed_right = right
            missing = 0

    follow(range(seed_y - 1, -1, -1))
    follow(range(seed_y + 1, height))
    return traced


def recover_undertraced_road_component(
    candidate: np.ndarray,
    traced: np.ndarray,
) -> np.ndarray:
    """Recover a large, central component when corridor tracing over-shrinks.

    Recovery is deliberately limited to pixels already present in the
    post-terrain candidate.  A component must have meaningful area and touch
    the lower central support band; isolated sky/building blobs therefore do
    not get reintroduced.
    """
    if candidate.shape != traced.shape:
        raise ValueError("candidate and traced masks must have the same shape")
    binary = (candidate > 0).astype(np.uint8)
    traced_binary = traced > 0
    candidate_area = int(np.count_nonzero(binary))
    traced_area = int(np.count_nonzero(traced_binary))
    if candidate_area == 0:
        return traced

    height, width = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    recovered = traced.copy()

    # Intersections and zebra crossings can split one model road prediction
    # into two bottom components. Recover only a sizeable component that is
    # vertically aligned and immediately adjacent to the traced corridor.
    if traced_area:
        traced_y, traced_x = np.nonzero(traced_binary)
        trace_top = int(traced_y.min())
        trace_bottom = int(traced_y.max())
        trace_left = int(traced_x.min())
        trace_right = int(traced_x.max())
        min_adjacent_area = max(64, int(round(height * width * 0.012)))
        max_horizontal_gap = max(4, int(round(width * 0.025)))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            component = labels == label
            if np.any(component & traced_binary) or area < min_adjacent_area:
                continue
            component_bottom = y + component_height - 1
            if component_bottom < int(round(height * 0.90)):
                continue
            vertical_overlap = max(
                0,
                min(component_bottom, trace_bottom) - max(y, trace_top) + 1,
            )
            if vertical_overlap < min(component_height, trace_bottom - trace_top + 1) * 0.5:
                continue
            component_right = x + component_width - 1
            horizontal_gap = max(
                0,
                trace_left - component_right - 1,
                x - trace_right - 1,
            )
            if horizontal_gap <= max_horizontal_gap:
                recovered[component] = 255

    if traced_area and traced_area >= candidate_area * 0.45:
        return recovered

    min_area = max(64, int(round(height * width * 0.03)))
    valid_bottom = height - 1
    # Respect an already-applied ROI (for example, a fixed hood exclusion).
    rows = np.flatnonzero(np.any(binary > 0, axis=1))
    if rows.size:
        valid_bottom = int(rows[-1])
    support_start = max(0, int(round(valid_bottom * 0.72)))
    centre_left = int(round(width * 0.30))
    centre_right = int(round(width * 0.70))
    selected: list[int] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < min_area or y + component_height - 1 < support_start:
            continue
        component = labels == label
        support = component[support_start : valid_bottom + 1, centre_left:centre_right]
        if np.count_nonzero(support) < max(8, int(round(width * 0.01))):
            continue
        selected.append(label)
    if not selected:
        return recovered

    for label in selected:
        recovered[labels == label] = 255
    # Conservative union: tracing remains intact and no absent pixels are added.
    return cv2.bitwise_or(traced, recovered)


@dataclass
class TemporalMaskFilter:
    """Reject short catastrophic area drops in an otherwise stable mask."""

    drop_ratio: float = 0.45
    max_fallback_frames: int = 2
    previous: np.ndarray | None = None
    fallback_frames: int = 0
    used_fallback: bool = False

    def reset(self) -> None:
        self.previous = None
        self.fallback_frames = 0
        self.used_fallback = False

    def update(self, mask: np.ndarray) -> np.ndarray:
        self.used_fallback = False
        if self.previous is not None:
            previous_area = np.count_nonzero(self.previous)
            current_area = np.count_nonzero(mask)
            dropped = (
                previous_area > 0
                and current_area < previous_area * self.drop_ratio
            )
            if dropped and self.fallback_frames < self.max_fallback_frames:
                self.fallback_frames += 1
                self.used_fallback = True
                return self.previous.copy()

        self.fallback_frames = 0
        self.previous = mask.copy()
        return mask


_SCENE_SIGNATURE_SIZE = (64, 36)
_SCENE_STRUCTURE_MAD_THRESHOLD = 32.0
_SCENE_STRONG_STRUCTURE_MAD_THRESHOLD = 65.0
_SCENE_STRONG_INTENSITY_MAD_THRESHOLD = 180.0
_SCENE_COLOR_DISTANCE_THRESHOLD = 0.30


def _frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(frame_bgr, _SCENE_SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)


def _signatures_show_scene_cut(previous: np.ndarray, current: np.ndarray) -> bool:
    previous_raw_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_raw_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    intensity_mad = float(np.mean(cv2.absdiff(previous_raw_gray, current_raw_gray)))
    previous_gray = cv2.equalizeHist(previous_raw_gray)
    current_gray = cv2.equalizeHist(current_raw_gray)
    structure_mad = float(np.mean(cv2.absdiff(previous_gray, current_gray)))

    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    previous_hist = cv2.calcHist([previous_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    current_hist = cv2.calcHist([current_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(previous_hist, previous_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    color_distance = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
    return bool(
        intensity_mad >= _SCENE_STRONG_INTENSITY_MAD_THRESHOLD
        or structure_mad >= _SCENE_STRONG_STRUCTURE_MAD_THRESHOLD
        or (
            structure_mad >= _SCENE_STRUCTURE_MAD_THRESHOLD
            and color_distance >= _SCENE_COLOR_DISTANCE_THRESHOLD
        )
    )


def is_scene_cut(previous_bgr: np.ndarray, current_bgr: np.ndarray) -> bool:
    """Return whether two frames are visually unrelated, independent of scale."""
    return _signatures_show_scene_cut(
        _frame_signature(previous_bgr),
        _frame_signature(current_bgr),
    )


@dataclass(frozen=True)
class LaneModel:
    """Line model expressed as x = slope * y + intercept."""

    slope: float
    intercept: float
    y_min: int
    y_max: int


def _cluster_lane_models(
    models: list[LaneModel],
    width: int,
) -> list[LaneModel]:
    if not models:
        return []

    reference_y = max(model.y_max for model in models)
    ordered = sorted(
        models,
        key=lambda model: model.slope * reference_y + model.intercept,
    )
    groups: list[list[LaneModel]] = []
    threshold = width * 0.06
    for model in ordered:
        bottom_x = model.slope * reference_y + model.intercept
        if not groups:
            groups.append([model])
            continue

        previous = groups[-1]
        previous_x = float(
            np.median(
                [
                    item.slope * reference_y + item.intercept
                    for item in previous
                ]
            )
        )
        if abs(bottom_x - previous_x) <= threshold:
            previous.append(model)
        else:
            groups.append([model])

    return [
        LaneModel(
            slope=float(np.median([item.slope for item in group])),
            intercept=float(np.median([item.intercept for item in group])),
            y_min=min(item.y_min for item in group),
            y_max=max(item.y_max for item in group),
        )
        for group in groups
    ]


def fit_lane_boundaries(
    candidates: np.ndarray,
    horizon_ratio: float = 0.35,
) -> tuple[np.ndarray, list[LaneModel]]:
    """Fit continuous longitudinal boundaries to fragmented candidates."""
    if candidates.ndim != 2:
        raise ValueError("candidates must be a two-dimensional array")

    height, width = candidates.shape
    lines = cv2.HoughLinesP(
        (candidates > 0).astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180,
        threshold=22,
        minLineLength=max(16, height // 12),
        maxLineGap=max(12, height // 15),
    )
    models: list[LaneModel] = []
    if lines is None:
        return np.zeros_like(candidates), models

    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dy = y2 - y1
        dx = x2 - x1
        if abs(dy) < max(10, abs(dx) * 0.45):
            continue
        slope = (x2 - x1) / float(y2 - y1)
        intercept = x1 - slope * y1
        if abs(slope) < 0.05:
            continue
        models.append(
            LaneModel(
                slope=float(slope),
                intercept=float(intercept),
                y_min=min(y1, y2),
                y_max=max(y1, y2),
            )
        )

    models = _cluster_lane_models(models, width)
    output = np.zeros_like(candidates)
    top = int(round(height * horizon_ratio))
    for model in models:
        y1 = max(top, model.y_min)
        y2 = min(height - 1, max(model.y_max, int(height * 0.70)))
        if y2 <= y1:
            continue
        x1 = int(np.clip(model.slope * y1 + model.intercept, 0, width - 1))
        x2 = int(np.clip(model.slope * y2 + model.intercept, 0, width - 1))
        thickness = max(4, width // 240)
        cv2.line(output, (x1, y1), (x2, y2), 255, thickness)
    return output, models


def render_hough_segments(candidates: np.ndarray, roi_top_ratio: float = 0.4) -> np.ndarray:
    """Render the original per-segment Hough result without extrapolating lines."""
    height, width = candidates.shape
    roi_top = int(round(height * roi_top_ratio))
    edges = cv2.Canny(candidates, 30, 100)
    roi_edges = edges[roi_top:, :]
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=30,
        maxLineGap=50,
    )
    output = np.zeros_like(candidates)
    if lines is None:
        return output
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in line)
        dy = y2 - y1
        dx = x2 - x1
        # Fence/guardrail strokes are mostly horizontal and sit high in the
        # image. Keep only elongated, road-directed segments reaching into the
        # lower scene; arrows remain eligible because they are vertical.
        if abs(dy) < max(15, abs(dx) * 0.25):
            continue
        if max(y1, y2) + roi_top < int(height * 0.58):
            continue
        cv2.line(output, (x1, y1 + roi_top), (x2, y2 + roi_top), 255, 2)
    return output


def render_original_hough_segments(
    candidates: np.ndarray,
    roi_top_ratio: float = 0.4,
) -> np.ndarray:
    """Match the original script's raw Canny + Hough rendering exactly."""
    height, _ = candidates.shape
    roi_top = int(round(height * roi_top_ratio))
    edges = cv2.Canny(candidates, 30, 100)
    lines = cv2.HoughLinesP(
        edges[roi_top:, :],
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=30,
        maxLineGap=50,
    )
    output = np.zeros_like(candidates)
    if lines is None:
        return output
    models = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in line)
        dy = y2 - y1
        if abs(dy) < 8:
            continue
        slope = (x2 - x1) / float(dy)
        intercept = x1 - slope * y1
        length = float(np.hypot(x2 - x1, y2 - y1))
        models.append([slope, intercept, min(y1, y2), max(y1, y2), length])
    # Hough sees both painted edges of one stripe. Merge near-parallel lines
    # before drawing so a single lane marking is not rendered twice.
    models.sort(key=lambda item: item[0])
    clusters = []
    for model in models:
        slope, intercept, y_min, y_max, length = model
        y_ref = height * 0.72 - roi_top
        x_ref = slope * y_ref + intercept
        best = None
        for cluster in clusters:
            if abs(slope - cluster[0]) < 0.08 and abs(x_ref - cluster[5]) < 42:
                best = cluster
                break
        if best is None:
            clusters.append([slope, intercept, y_min, y_max, length, x_ref])
        else:
            total = best[4] + length
            best[0] = (best[0] * best[4] + slope * length) / total
            best[1] = (best[1] * best[4] + intercept * length) / total
            best[2] = min(best[2], y_min)
            best[3] = max(best[3], y_max)
            best[4] = total
            best[5] = best[0] * y_ref + best[1]
    for slope, intercept, y_min, y_max, _, _ in clusters:
        x1 = int(np.clip(slope * y_min + intercept, 0, candidates.shape[1] - 1))
        x2 = int(np.clip(slope * y_max + intercept, 0, candidates.shape[1] - 1))
        cv2.line(output, (x1, y_min + roi_top), (x2, y_max + roi_top), 255, 2)
    return output


def render_geometric_lane_boundaries(
    candidates: np.ndarray,
    road_mask: np.ndarray | None = None,
    horizon_ratio: float = 0.42,
    bottom_ratio: float = 0.68,
    min_endpoint_ratio: float = 0.55,
    project_to_bottom: bool = True,
    require_pair: bool = False,
    require_opposite_slopes: bool = False,
    single_line_only: bool = False,
) -> tuple[np.ndarray, int, float]:
    """Render at most one coherent boundary per side.

    Raw Hough output is not a lane detector: crosswalks, text, fences and
    reflections all produce line segments.  This renderer therefore requires
    a long, road-directed segment reaching the lower driving corridor, groups
    parallel segments, and keeps only the strongest left/right hypotheses.
    If the geometry is ambiguous it returns an empty mask (unknown), which is
    safer than displaying an invented lane boundary.
    """
    if candidates.ndim != 2:
        raise ValueError("candidates must be a two-dimensional array")
    height, width = candidates.shape
    if road_mask is not None and road_mask.shape != candidates.shape:
        raise ValueError("road_mask must have the same shape as candidates")

    source = (candidates > 0).astype(np.uint8) * 255
    if road_mask is not None:
        source = cv2.bitwise_and(source, road_mask)
    roi_top = int(round(height * horizon_ratio))
    roi_bottom = min(height - 1, int(round(height * bottom_ratio)))
    edges = cv2.Canny(source, 30, 100)
    edges[:roi_top] = 0
    edges[roi_bottom + 1 :] = 0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(14, width // 65),
        minLineLength=max(28, int(height * 0.075)),
        maxLineGap=max(12, int(height * 0.055)),
    )
    if lines is None:
        return np.zeros_like(candidates), 0, 0.0

    reference_y = float(roi_bottom)
    hypotheses: list[dict[str, float]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        # Normalize endpoints so y1 is the upper endpoint.
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dy = float(y2 - y1)
        dx = float(x2 - x1)
        length = float(np.hypot(dx, dy))
        if dy < height * 0.065 or length < height * 0.075:
            continue
        slope = dx / dy
        # Near-horizontal strokes are crosswalks/fences; extreme slopes are
        # usually vehicle edges or image artefacts.
        if abs(slope) < 0.10 or abs(slope) > 3.5:
            continue
        if y2 < height * min_endpoint_ratio:
            continue
        if project_to_bottom:
            x_at_ref = x1 + slope * (reference_y - y1)
            x_at_top = x1 + slope * (roi_top - y1)
        else:
            # For an occluded upper marking, use observed endpoints only;
            # projecting it to the hood would invent an unobserved boundary.
            x_at_ref = float(x2)
            x_at_top = float(x1)
        if not (-0.20 * width <= x_at_ref <= 1.20 * width):
            continue
        if not (-0.05 * width <= x_at_top <= 1.05 * width):
            continue
        hypotheses.append(
            {
                "slope": slope,
                "intercept": x1 - slope * y1,
                "x_ref": x_at_ref,
                "length": length,
                "y_min": float(y1),
                "y_max": float(y2),
            }
        )
    if not hypotheses:
        return np.zeros_like(candidates), 0, 0.0

    # Group duplicate edges of one stripe by their position at the bottom.
    hypotheses.sort(key=lambda item: item["x_ref"])
    groups: list[list[dict[str, float]]] = []
    for item in hypotheses:
        if not groups or item["x_ref"] - groups[-1][-1]["x_ref"] > width * 0.07:
            groups.append([item])
        else:
            groups[-1].append(item)

    grouped: list[dict[str, float]] = []
    for group in groups:
        weights = np.array([item["length"] for item in group], dtype=np.float32)
        grouped.append(
            {
                "slope": float(np.average([item["slope"] for item in group], weights=weights)),
                "intercept": float(np.average([item["intercept"] for item in group], weights=weights)),
                "x_ref": float(np.average([item["x_ref"] for item in group], weights=weights)),
                "length": float(weights.sum()),
                "y_min": min(item["y_min"] for item in group),
                "y_max": max(item["y_max"] for item in group),
            }
        )

    centre = width * 0.50
    selected: list[dict[str, float]] = []
    if single_line_only:
        # Explicit night fallback: retain only the strongest valid observed
        # hypothesis, without inventing an opposite boundary.
        selected.append(max(grouped, key=lambda item: item["length"]))
    else:
        left = [item for item in grouped if item["x_ref"] < centre + width * 0.06]
        right = [item for item in grouped if item["x_ref"] > centre - width * 0.06]
        if left:
            selected.append(max(left, key=lambda item: item["length"]))
        if right:
            best = max(right, key=lambda item: item["length"])
            if not selected or abs(best["x_ref"] - selected[0]["x_ref"]) > width * 0.10:
                selected.append(best)

    # A pair is only meaningful when it describes a left and right corridor
    # boundary.  This check applies even in the permissive single-line mode:
    # otherwise two same-direction or crossing hypotheses can be rendered as
    # an invented X-shaped lane.  Keep single hypotheses eligible for recall.
    if len(selected) >= 2:
        left_item, right_item = selected[0], selected[1]
        overlap_top = max(
            float(roi_top),
            left_item["y_min"],
            right_item["y_min"],
        )
        overlap_bottom = min(
            float(roi_bottom),
            left_item["y_max"],
            right_item["y_max"],
        )
        if (
            overlap_bottom <= overlap_top
            or left_item["slope"] >= -0.04
            or right_item["slope"] <= 0.04
        ):
            return np.zeros_like(candidates), 0, 0.0
        left_top = left_item["slope"] * overlap_top + left_item["intercept"]
        right_top = right_item["slope"] * overlap_top + right_item["intercept"]
        left_bottom = left_item["slope"] * overlap_bottom + left_item["intercept"]
        right_bottom = right_item["slope"] * overlap_bottom + right_item["intercept"]
        if (
            left_top >= right_top - width * 0.01
            or left_bottom >= right_bottom - width * 0.01
        ):
            return np.zeros_like(candidates), 0, 0.0

    if require_pair:
        if len(selected) != 2:
            return np.zeros_like(candidates), 0, 0.0
        if require_opposite_slopes and not (
            selected[0]["slope"] < -0.04
            and selected[1]["slope"] > 0.04
        ):
            return np.zeros_like(candidates), 0, 0.0
        overlap_top = max(
            float(roi_top),
            selected[0]["y_min"],
            selected[1]["y_min"],
        )
        left_top = (
            selected[0]["slope"] * overlap_top
            + selected[0]["intercept"]
        )
        right_top = (
            selected[1]["slope"] * overlap_top
            + selected[1]["intercept"]
        )
        if left_top >= right_top - width * 0.01:
            return np.zeros_like(candidates), 0, 0.0
    if not selected:
        return np.zeros_like(candidates), 0, 0.0

    output = np.zeros_like(candidates)
    support_values: list[float] = []
    thickness = max(2, width // 420)
    for item in selected:
        y1 = max(roi_top, int(item["y_min"]))
        y2 = min(roi_bottom, int(item["y_max"]))
        if y2 - y1 < height * 0.08:
            continue
        x1 = int(np.clip(item["slope"] * y1 + item["intercept"], 0, width - 1))
        x2 = int(np.clip(item["slope"] * y2 + item["intercept"], 0, width - 1))
        cv2.line(output, (x1, y1), (x2, y2), 255, thickness)
        support_values.append(min(1.0, item["length"] / (height * 0.35)))
    count = int(np.count_nonzero(output) > 0)
    confidence = float(np.mean(support_values)) if support_values else 0.0
    if single_line_only:
        confidence = min(confidence, 0.65)
    return output, count, confidence


def smooth_binary_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Close small gaps and remove staircase edges while keeping a binary mask."""
    size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (size, size), 0)
    return (blurred >= 127).astype(np.uint8) * 255


def smooth_road_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth stair-stepped semantic edges while retaining the lower road."""
    height, width = mask.shape
    size = max(9, int(round(min(height, width) / 55)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (size * 2 + 1, size * 2 + 1), 0)
    return (blurred >= 127).astype(np.uint8) * 255


def suppress_peripheral_terrain(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
) -> np.ndarray:
    """Remove large side-connected snow or vegetation from a road mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    snow = cv2.inRange(
        hsv,
        np.array([0, 0, 175], dtype=np.uint8),
        np.array([180, 55, 255], dtype=np.uint8),
    )
    vegetation = cv2.inRange(
        hsv,
        np.array([28, 45, 25], dtype=np.uint8),
        np.array([95, 255, 220], dtype=np.uint8),
    )
    terrain = cv2.bitwise_or(snow, vegetation)
    terrain = cv2.morphologyEx(
        terrain,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    height, width = road_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (terrain > 0).astype(np.uint8),
        connectivity=8,
    )
    rejected = np.zeros_like(road_mask)
    min_area = max(64, int(round(height * width * 0.003)))
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_side = x <= 2 or x + component_width >= width - 2
        reaches_camera = y + component_height >= int(height * 0.90)
        if not touches_side or area < min_area:
            continue

        component = labels == label
        snow_fraction = np.count_nonzero(snow[component]) / float(area)
        is_snow = snow_fraction >= 0.5
        centre_x = x + component_width / 2.0
        in_side_band = centre_x <= width * 0.25 or centre_x >= width * 0.75
        # Snow banks can run all the way to the camera, unlike the smaller
        # peripheral vegetation components handled by the legacy reach test.
        # Keep central snow-covered roadway components while rejecting only
        # sizeable, clearly lateral snow masses.
        if (is_snow and in_side_band) or (not is_snow and not reaches_camera):
            rejected[labels == label] = 255
    return cv2.bitwise_and(road_mask, cv2.bitwise_not(rejected))


@dataclass
class LaneTracker:
    """Keep trustworthy lane models through very short occlusions."""

    max_missing_frames: int = 2
    previous: list | None = None
    missing_frames: int = 0

    def reset(self) -> None:
        self.previous = None
        self.missing_frames = 0

    def update(self, current: list) -> list:
        if current:
            self.previous = current
            self.missing_frames = 0
            return current
        if self.previous is not None and self.missing_frames < self.max_missing_frames:
            self.missing_frames += 1
            return self.previous
        self.previous = None
        return []


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    road_model_path: str | None = None
    road_class_id: int | None = None
    road_probability_threshold: float = 0.5
    # A lower threshold from the same binary road logits may be used only as
    # lane-search support. It never changes the returned/painted road mask.
    lane_road_probability_threshold: float | None = None
    device: str = "cuda"
    inference_height: int = 512
    inference_width: int = 896
    hood_top_ratio: float = 0.68
    # A fixed crop is only valid after camera-specific hood calibration.
    # Keep it opt-in: otherwise the lower scene must remain observable.
    fixed_hood_crop: bool = False
    # Lane search is deliberately independent from the hood/overlay crop.
    # The old implementation used hood_top_ratio for both and discarded the
    # entire lower scene.  This bound is only the renderer's safe image edge;
    # it is not a hood detector.
    lane_bottom_ratio: float = 0.96
    road_alpha: float = 0.42
    max_lane_history: int = 2
    lane_model_path: str | None = None
    lane_probability_threshold: float = 0.65
    road_inference_stride: int = 1
    lane_inference_stride: int = 1
    # The raw HSV+Canny+Hough path is retained for historical VIL100
    # comparison, but is never the production default because it produces
    # arbitrary lines on BDD/night/snow scenes.
    legacy_lane_mode: bool = False
    # Broad horizontal paint is opt-in: it can otherwise turn glare and road
    # texture into false lane detections.
    enable_surface_markings: bool = False
    # Emit only pixels supported by observed image/model candidates; never
    # extrapolate fitted lane geometry into unobserved regions.
    observed_markings_only: bool = False
    # Low-light enhancement is retained for experiments but disabled by
    # default to preserve the stable raw-frame lane inference behavior.
    enable_low_light_enhancement: bool = False
    # Added after v11 for experimental night recovery.  It can invent an
    # isolated boundary from one reflection, so the v11 safe profile keeps it
    # disabled unless explicitly requested.
    enable_night_single_line_recovery: bool = False

    def __post_init__(self) -> None:
        if self.road_class_id is not None and self.road_class_id not in (0, 1):
            raise ValueError("road_class_id must be 0 or 1")


@dataclass(frozen=True)
class FrameDiagnostics:
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    lane_count: int
    road_area_ratio: float
    used_road_fallback: bool
    lane_confidence: float = 0.0
    lane_rejected: bool = False


@dataclass(frozen=True)
class DetectionResult:
    road_mask: np.ndarray
    lane_mask: np.ndarray
    overlay: np.ndarray
    diagnostics: FrameDiagnostics


def observed_markings_mask(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    learned_candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Return a conservative mask of actually observed marking pixels.

    The observed-fixed behavior combines the learned lane candidates with
    strict white/yellow paint candidates, clips the union to the road, and
    removes only broad lower-image hood/reflection components.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image")
    if road_mask.ndim != 2 or road_mask.shape != frame_bgr.shape[:2]:
        raise ValueError("road_mask must match frame dimensions")
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180], np.uint8), np.array([180, 55, 255], np.uint8))
    yellow = cv2.inRange(hsv, np.array([15, 70, 90], np.uint8), np.array([40, 255, 255], np.uint8))
    paint = cv2.bitwise_or(white, yellow)
    source = paint
    if learned_candidates is not None:
        if learned_candidates.shape != road_mask.shape:
            raise ValueError("learned_candidates must match frame dimensions")
        learned = (learned_candidates > 0).astype(np.uint8) * 255
        source = cv2.bitwise_or(source, learned)
    source = cv2.bitwise_and(source, road_mask)
    cleaned = cv2.morphologyEx(source, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # A glossy hood/reflection often forms a large bright blob touching the
    # bottom border. Remove only those geometrically implausible blobs; do
    # not crop the lower half, since genuine nearby lane paint can be there.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8), 8
    )
    height, width = cleaned.shape
    for label in range(1, count):
        x, y, box_w, box_h, area = stats[label]
        touches_bottom = y + box_h >= height - 1
        broad_or_large = (
            box_w >= max(48, int(width * 0.12))
            or (box_w >= int(width * 0.05) and area >= int(width * height * 0.012))
        )
        hood_like = (
            y >= int(height * 0.70)
            and box_w >= int(width * 0.08)
            and box_h >= int(height * 0.05)
        )
        if (touches_bottom and broad_or_large) or hood_like:
            cleaned[labels == label] = 0
    return cv2.bitwise_and(cleaned, source)


def _render_lane_models(
    shape: tuple[int, int],
    models: list[LaneModel],
    horizon_ratio: float,
    bottom_ratio: float,
) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width), dtype=np.uint8)
    y_top = int(round(height * horizon_ratio))
    y_bottom = min(height - 1, int(round(height * bottom_ratio)))
    top_thickness = max(2, width // 360)
    bottom_thickness = max(6, width // 80)
    for model in models:
        segment_count = max(1, min(32, (y_bottom - y_top) // 8))
        y_values = np.linspace(y_top, y_bottom, segment_count + 1).astype(int)
        for index, (start_y, end_y) in enumerate(zip(y_values, y_values[1:])):
            progress = (index + 0.5) / segment_count
            thickness = int(
                round(
                    top_thickness
                    + (bottom_thickness - top_thickness) * progress
                )
            )
            start_x = int(
                np.clip(model.slope * start_y + model.intercept, 0, width - 1)
            )
            end_x = int(
                np.clip(model.slope * end_y + model.intercept, 0, width - 1)
            )
            cv2.line(
                output,
                (start_x, start_y),
                (end_x, end_y),
                255,
                thickness,
            )
    return output


class RoadLaneDetector:
    """Reusable per-frame road and longitudinal lane detector."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        road_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
        lane_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self._road_predictor = road_predictor
        self._lane_predictor = lane_predictor
        self._processor = None
        self._model = None
        self._lane_model = None
        self._lane_height = self.config.inference_height
        self._lane_width = self.config.inference_width
        self._torch = None
        self._device = self.config.device
        self._road_class_id = resolve_road_class_id(
            self.config.road_class_id,
            self.config.road_model_path,
        )
        self._binary_road_model = False
        self._road_filter = TemporalMaskFilter()
        self._lane_road_filter = TemporalMaskFilter()
        self._previous_frame_signature: np.ndarray | None = None
        self._previous_frame_shape: tuple[int, int, int] | None = None
        self._lane_tracker = LaneTracker(self.config.max_lane_history)
        if self.config.road_inference_stride < 1:
            raise ValueError("road_inference_stride must be at least 1")
        if self.config.lane_inference_stride < 1:
            raise ValueError("lane_inference_stride must be at least 1")
        if not 0.0 < self.config.road_probability_threshold < 1.0:
            raise ValueError("road_probability_threshold must be between 0 and 1")
        lane_road_threshold = self.config.lane_road_probability_threshold
        if lane_road_threshold is not None and not (
            0.0 < lane_road_threshold <= self.config.road_probability_threshold
        ):
            raise ValueError(
                "lane_road_probability_threshold must be between 0 and "
                "road_probability_threshold"
            )
        if not 0.5 <= self.config.lane_probability_threshold < 1.0:
            raise ValueError("lane_probability_threshold must be in [0.5, 1.0)")
        if not 0.5 <= self.config.lane_bottom_ratio <= 1.0:
            raise ValueError("lane_bottom_ratio must be between 0.5 and 1.0")
        self._frame_index = 0
        self._cached_road_candidate: np.ndarray | None = None
        self._cached_lane_road_candidate: np.ndarray | None = None
        self._lane_frame_index = 0
        self._cached_lane_prediction: np.ndarray | None = None
        if self._road_predictor is None:
            self._load_segformer()
        if self._lane_predictor is None and self.config.lane_model_path:
            self._load_lane_model()

    def reset_temporal_state(self) -> None:
        """Clear all frame-dependent state before an unrelated sequence."""
        self._road_filter.reset()
        self._lane_road_filter.reset()
        self._lane_tracker.reset()
        self._previous_frame_signature = None
        self._previous_frame_shape = None
        self._cached_road_candidate = None
        self._cached_lane_road_candidate = None
        self._frame_index = 0
        self._cached_lane_prediction = None
        self._lane_frame_index = 0

    def _load_segformer(self) -> None:
        checkpoint_path = (
            Path(self.config.road_model_path)
            if self.config.road_model_path is not None
            else None
        )
        if checkpoint_path is not None and not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Road model checkpoint does not exist: {checkpoint_path}"
            )
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        requested = self.config.device
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self._device = requested
        self._torch = torch
        model_name = self.config.model_name
        checkpoint = None
        state_dict = None
        if checkpoint_path is not None:
            checkpoint = load_road_checkpoint(checkpoint_path, torch)
            if not isinstance(checkpoint, dict):
                raise ValueError("Road model checkpoint must be a state-dict mapping")
            model_name = checkpoint.get(
                "model_name",
                "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                raise ValueError("Road model checkpoint state_dict must be a mapping")
            self._road_class_id = resolve_road_class_id(
                self.config.road_class_id,
                self.config.road_model_path,
                checkpoint,
            )
            self._binary_road_model = True

        self._processor = load_pretrained_cached_first(
            SegformerImageProcessor,
            model_name,
            size={
                "height": self.config.inference_height,
                "width": self.config.inference_width,
            },
        )
        if state_dict is None:
            self._model = load_pretrained_cached_first(
                SegformerForSemanticSegmentation,
                model_name,
            )
        else:
            self._model = load_pretrained_cached_first(
                SegformerForSemanticSegmentation,
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            # Training uses a binary BDD head regardless of Cityscapes classes.
            self._model.decode_head.classifier = torch.nn.Conv2d(
                self._model.config.decoder_hidden_size,
                2,
                kernel_size=1,
            )
            try:
                self._model.load_state_dict(state_dict, strict=True)
            except RuntimeError as error:
                raise ValueError(
                    f"Road model checkpoint is incompatible with {model_name}: {error}"
                ) from error
        self._model = self._model.to(self._device)
        self._model.eval()

    def _predict_road_candidates(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._road_predictor is not None:
            predicted = self._road_predictor(frame_bgr)
            if predicted.shape != frame_bgr.shape[:2]:
                raise ValueError("road predictor returned a mask with the wrong shape")
            road = (predicted > 0).astype(np.uint8) * 255
            return road, road

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        if self._binary_road_model:
            probability = self._torch.softmax(logits, dim=1)[0, self._road_class_id]
            probability_np = probability.cpu().numpy()
            road = road_mask_from_probability(
                probability_np,
                self.config.road_probability_threshold,
            )
            lane_road_threshold = self.config.lane_road_probability_threshold
            lane_road = (
                road_mask_from_probability(probability_np, lane_road_threshold)
                if lane_road_threshold is not None
                else road
            )
        else:
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            road = road_mask_from_prediction(prediction, self._road_class_id)
            lane_road = road
        size = (frame_bgr.shape[1], frame_bgr.shape[0])
        road = cv2.resize(
            road,
            size,
            interpolation=cv2.INTER_NEAREST,
        )
        lane_road = cv2.resize(lane_road, size, interpolation=cv2.INTER_NEAREST)
        return road, lane_road

    def _predict_road(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the final road candidate; retained for private compatibility."""
        return self._predict_road_candidates(frame_bgr)[0]

    def _load_lane_model(self) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation

        checkpoint = torch.load(
            self.config.lane_model_path,
            map_location="cpu",
            weights_only=True,
        )
        model_name = checkpoint.get("model_name", self.config.model_name)
        self._lane_height = self.config.inference_height
        self._lane_width = self.config.inference_width
        model = load_pretrained_cached_first(
            SegformerForSemanticSegmentation,
            model_name,
        )
        model.decode_head.classifier = torch.nn.Conv2d(
            model.config.decoder_hidden_size,
            2,
            kernel_size=1,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        self._lane_model = model
        self._torch = torch

    def _predict_lane(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._lane_predictor is not None:
            predicted = self._lane_predictor(frame_bgr)
            if predicted.shape != frame_bgr.shape[:2]:
                raise ValueError("lane predictor returned a mask with the wrong shape")
            return (predicted > 0).astype(np.uint8) * 255
        if self._lane_model is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self._lane_width, self._lane_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        pixel_values = (
            self._torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        with self._torch.inference_mode():
            logits = self._lane_model(pixel_values=pixel_values).logits
            logits = self._torch.nn.functional.interpolate(
                logits,
                size=(self._lane_height, self._lane_width),
                mode="bilinear",
                align_corners=False,
            )
        probability = self._torch.softmax(logits, dim=1)[0, 1]
        lane = (
            probability >= self.config.lane_probability_threshold
        ).cpu().numpy().astype(np.uint8) * 255
        return cv2.resize(
            lane,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _lane_candidates(
        frame_bgr: np.ndarray,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.bitwise_and(frame_rgb, frame_rgb, mask=road_mask)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 30, 255], dtype=np.uint8),
        )
        yellow = cv2.inRange(
            hsv,
            np.array([15, 80, 100], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )

        candidates = cv2.bitwise_or(white, yellow)
        candidates = cv2.morphologyEx(
            candidates,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        candidates = cv2.morphologyEx(
            candidates,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
        )
        return candidates

    @staticmethod
    def _lane_gradient_candidates(
        frame_bgr: np.ndarray,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        """Find low-saturation lane paint visible at night or in glare.

        The color detector intentionally stays conservative.  This separate
        gradient channel is only used after it is intersected with the learned
        lane mask, so road texture cannot become an unconstrained Hough input.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 45, 130)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        return cv2.bitwise_and(edges, road_mask)

    @staticmethod
    def _reflective_lane_candidates(
        frame_bgr: np.ndarray,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        """Extract narrow locally bright markings, excluding broad light glow."""
        lightness = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        enhanced = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8),
        ).apply(lightness)
        top_hat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        )
        values = top_hat[road_mask > 0]
        if values.size == 0:
            return np.zeros_like(road_mask)
        threshold = int(np.clip(np.percentile(values, 90), 18, 55))
        candidates = ((top_hat >= threshold) & (enhanced >= 70)).astype(np.uint8) * 255
        candidates = cv2.bitwise_and(candidates, road_mask)
        return cv2.morphologyEx(
            candidates,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

    @staticmethod
    def _postprocess_road_candidate(
        frame_bgr: np.ndarray,
        candidate: np.ndarray,
        camera_roi: np.ndarray,
        temporal_filter: TemporalMaskFilter,
    ) -> np.ndarray:
        height, width = candidate.shape
        road_mask = cv2.bitwise_and(candidate, camera_roi)
        kernel_size = max(5, int(round(min(height, width) / 140)) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        road_mask = suppress_peripheral_terrain(frame_bgr, road_mask)
        suppressed_candidate = road_mask
        road_mask = trace_road_corridor(suppressed_candidate)
        road_mask = recover_undertraced_road_component(
            suppressed_candidate,
            road_mask,
        )
        road_mask = cv2.bitwise_and(road_mask, camera_roi)
        road_mask = temporal_filter.update(road_mask)
        road_mask = cv2.bitwise_and(road_mask, camera_roi)
        road_mask = smooth_road_mask(road_mask)
        road_mask = cv2.bitwise_and(road_mask, camera_roi)
        road_mask[camera_roi == 0] = 0
        return road_mask

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a BGR uint8 image with three channels")
        if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            raise ValueError("frame dimensions must be non-empty")

        started = perf_counter()
        frame_shape = frame_bgr.shape
        if self._previous_frame_shape is not None and self._previous_frame_shape != frame_shape:
            self.reset_temporal_state()
        current_signature = _frame_signature(frame_bgr)
        if (
            self._previous_frame_signature is not None
            and _signatures_show_scene_cut(self._previous_frame_signature, current_signature)
        ):
            self.reset_temporal_state()
        self._previous_frame_signature = current_signature
        self._previous_frame_shape = frame_shape
        should_infer_road = (
            self._cached_road_candidate is None
            or self._frame_index % self.config.road_inference_stride == 0
        )
        if should_infer_road:
            (
                self._cached_road_candidate,
                self._cached_lane_road_candidate,
            ) = self._predict_road_candidates(frame_bgr)
        road_candidate = self._cached_road_candidate
        lane_road_candidate = self._cached_lane_road_candidate
        self._frame_index += 1
        inference_finished = perf_counter()

        height, width = frame_bgr.shape[:2]
        camera_roi = (
            build_camera_roi((height, width), self.config.hood_top_ratio)
            if self.config.fixed_hood_crop
            else np.full((height, width), 255, dtype=np.uint8)
        )
        road_mask = self._postprocess_road_candidate(
            frame_bgr,
            road_candidate,
            camera_roi,
            self._road_filter,
        )
        if self.config.lane_road_probability_threshold is None:
            lane_road_mask = road_mask
        else:
            lane_road_mask = self._postprocess_road_candidate(
                frame_bgr,
                lane_road_candidate,
                camera_roi,
                self._lane_road_filter,
            )
        # Decide this before lane inference so the learned predictor receives
        # a low-light view while all classical candidates retain the original
        # frame (avoiding headlight/glare line artifacts).
        night_mode = is_low_light_scene(frame_bgr, road_mask)
        surface_markings = (
            extract_surface_markings(frame_bgr, road_mask)
            if self.config.enable_surface_markings
            else np.zeros_like(road_mask, dtype=np.uint8)
        )

        lane_search_kernel_size = min(
            9,
            max(3, int(round(min(height, width) / 240)) | 1),
        )
        lane_search_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (lane_search_kernel_size, lane_search_kernel_size),
        )
        lane_scene_mask = cv2.dilate(lane_road_mask, lane_search_kernel)

        if self._lane_predictor is not None or self._lane_model is not None:
            should_infer_lane = (
                self._cached_lane_prediction is None
                or self._lane_frame_index % self.config.lane_inference_stride == 0
            )
            if should_infer_lane:
                lane_input = (
                    enhance_low_light_frame(frame_bgr)
                    if night_mode and self.config.enable_low_light_enhancement
                    else frame_bgr
                )
                self._cached_lane_prediction = self._predict_lane(lane_input)
            self._lane_frame_index += 1
            learned_candidates = cv2.bitwise_and(
                self._cached_lane_prediction,
                lane_scene_mask,
            )
            learned_candidates = cv2.morphologyEx(
                learned_candidates,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            )
            # The learned mask is intentionally used as a robust candidate mask,
            # while the final visualization is fitted to lane-shaped line models.
            # This prevents broad semantic blobs from being painted as lane lines.
            classical_candidates = self._lane_candidates(
                frame_bgr,
                lane_scene_mask,
            )
            lane_confidence = 0.0
            lane_rejected = False
            if self._lane_predictor is not None and self._lane_model is None:
                # An explicitly injected predictor is part of the public API;
                # preserve its pixel mask for downstream integration.  The
                # bundled HSV/Hough fallback below remains geometry-gated.
                lane_mask = learned_candidates
                visual_lane_mask = learned_candidates
                tracked_lane_count = int(np.count_nonzero(lane_mask) > 0)
                lane_confidence = 1.0 if tracked_lane_count else 0.0
                lane_rejected = tracked_lane_count == 0
            elif self._lane_model is not None and self.config.legacy_lane_mode:
                classical_candidates = self._lane_candidates(frame_bgr, lane_road_mask)
                lane_mask = render_original_hough_segments(classical_candidates)
                visual_lane_mask = lane_mask
                tracked_lane_count = int(np.count_nonzero(lane_mask) > 0)
            elif self._lane_model is not None:
                gradient_candidates = self._lane_gradient_candidates(
                    frame_bgr,
                    lane_scene_mask,
                )
                learned_gradient = cv2.bitwise_and(
                    gradient_candidates,
                    learned_candidates,
                )
                learned_pixels = int(np.count_nonzero(learned_candidates))
                require_reflection_pair = False
                if night_mode:
                    if learned_pixels < 500:
                        combined_candidates = self._reflective_lane_candidates(
                            frame_bgr,
                            lane_scene_mask,
                        )
                        require_reflection_pair = True
                    else:
                        learned_guided_classical = cv2.bitwise_and(
                            classical_candidates,
                            learned_candidates,
                        )
                        reflective_candidates = self._reflective_lane_candidates(
                            frame_bgr,
                            lane_scene_mask,
                        )
                        reflective_candidates = cv2.bitwise_and(
                            reflective_candidates,
                            cv2.dilate(
                                learned_candidates,
                                np.ones((7, 7), dtype=np.uint8),
                            ),
                        )
                        combined_candidates = cv2.bitwise_or(
                            learned_guided_classical,
                            learned_gradient,
                        )
                        combined_candidates = cv2.bitwise_or(
                            combined_candidates,
                            reflective_candidates,
                        )
                else:
                    combined_candidates = cv2.bitwise_or(
                        classical_candidates,
                        learned_gradient,
                    )
                lane_mask, tracked_lane_count, lane_confidence = (
                    render_geometric_lane_boundaries(
                        combined_candidates,
                        road_mask=lane_scene_mask,
                        horizon_ratio=0.42,
                        bottom_ratio=self.config.lane_bottom_ratio,
                        require_pair=require_reflection_pair,
                        require_opposite_slopes=require_reflection_pair,
                    )
                )
                # If the learned mask contains a coherent upper segment but
                # the lower road is occluded, preserve only that short segment
                # rather than extrapolating it into the vehicle path. This
                # recovers night/occluded labels without re-enabling raw Hough.
                if (
                    tracked_lane_count == 0
                    and np.count_nonzero(classical_candidates)
                    < max(1, int(np.count_nonzero(lane_road_mask) * 0.02))
                ):
                    lane_mask, tracked_lane_count, lane_confidence = (
                        render_geometric_lane_boundaries(
                            learned_candidates,
                            road_mask=lane_scene_mask,
                            horizon_ratio=0.30,
                            bottom_ratio=self.config.lane_bottom_ratio,
                            min_endpoint_ratio=0.34,
                            project_to_bottom=False,
                        )
                    )
                # A genuine night marking is often visible on only one side.
                # Retry once with pair requirements disabled, but keep the
                # learned/reflective/gradient gate and render only the
                # strongest observed segment (no projection into the hood).
                if (
                    self.config.enable_night_single_line_recovery
                    and night_mode
                    and tracked_lane_count == 0
                    and np.any(combined_candidates)
                ):
                    lane_mask, tracked_lane_count, lane_confidence = (
                        render_geometric_lane_boundaries(
                            combined_candidates,
                            road_mask=lane_scene_mask,
                            horizon_ratio=0.42,
                            bottom_ratio=self.config.lane_bottom_ratio,
                            project_to_bottom=False,
                            require_pair=False,
                            require_opposite_slopes=False,
                            single_line_only=True,
                        )
                    )
                    # Single-line evidence is intentionally lower confidence
                    # than a converging pair, even when its support is long.
                    lane_confidence = min(lane_confidence, 0.65)
                # Ambiguous geometry is explicitly represented as unknown.
                # Do not pass the broad semantic blob to the consumer.
                lane_rejected = tracked_lane_count == 0
                visual_lane_mask = lane_mask
            else:
                lane_mask, tracked_lane_count, lane_confidence = (
                    render_geometric_lane_boundaries(
                        classical_candidates,
                        road_mask=lane_scene_mask,
                        horizon_ratio=0.42,
                        bottom_ratio=self.config.lane_bottom_ratio,
                    )
                )
                visual_lane_mask = lane_mask
                lane_rejected = tracked_lane_count == 0
            if self.config.observed_markings_only and self._lane_model is not None:
                # Geometry fitting is useful for legacy mode, but its
                # projected lines are not observations. Keep only source
                # pixels that are present in the frame/candidate masks.
                observed = observed_markings_mask(
                    frame_bgr,
                    lane_road_mask,
                    learned_candidates,
                )
                lane_mask = observed
                visual_lane_mask = observed
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (observed > 0).astype(np.uint8), 8
                )
                tracked_lane_count = sum(
                    1 for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 2
                )
                support = np.count_nonzero(observed)
                road_support = max(1, np.count_nonzero(lane_road_mask))
                lane_confidence = min(1.0, support / max(1.0, road_support * 0.02))
                lane_rejected = tracked_lane_count == 0
            lane_mask = cv2.bitwise_and(lane_mask, lane_road_mask)
            visual_lane_mask = cv2.bitwise_and(visual_lane_mask, lane_road_mask)
        else:
            candidates = self._lane_candidates(frame_bgr, lane_road_mask)
            lane_mask, tracked_lane_count, lane_confidence = (
                render_geometric_lane_boundaries(
                    candidates,
                    road_mask=lane_scene_mask,
                    horizon_ratio=0.42,
                    bottom_ratio=self.config.lane_bottom_ratio,
                )
            )
            lane_mask = cv2.bitwise_and(lane_mask, lane_road_mask)
            visual_lane_mask = lane_mask
            lane_rejected = tracked_lane_count == 0
            if self.config.observed_markings_only:
                observed = observed_markings_mask(frame_bgr, lane_road_mask, candidates)
                lane_mask = observed
                visual_lane_mask = observed
                count, _, stats, _ = cv2.connectedComponentsWithStats(
                    (observed > 0).astype(np.uint8), 8
                )
                tracked_lane_count = sum(
                    1 for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 2
                )
                lane_confidence = min(
                    1.0,
                    np.count_nonzero(observed)
                    / max(1.0, np.count_nonzero(lane_road_mask) * 0.02),
                )
                lane_rejected = tracked_lane_count == 0

        # Surface markings are an independent branch: they enrich the pixel
        # output without entering longitudinal model fitting/statistics.
        if np.any(surface_markings) and not self.config.observed_markings_only:
            lane_mask = cv2.bitwise_or(lane_mask, surface_markings)
            visual_lane_mask = cv2.bitwise_or(visual_lane_mask, surface_markings)
            if tracked_lane_count == 0:
                tracked_lane_count = 1
                lane_confidence = min(lane_confidence, 0.35)
                lane_rejected = False

        if not np.any(lane_mask):
            tracked_lane_count = 0
            lane_confidence = 0.0
            lane_rejected = True

        overlay = frame_bgr.copy()
        road_pixels = road_mask > 0
        if np.any(road_pixels):
            green = np.array([0, 190, 0], dtype=np.float32)
            blended = (
                overlay[road_pixels].astype(np.float32)
                * (1.0 - self.config.road_alpha)
                + green * self.config.road_alpha
            )
            overlay[road_pixels] = np.clip(blended, 0, 255).astype(np.uint8)
        overlay[visual_lane_mask > 0] = np.array([0, 255, 255], dtype=np.uint8)

        finished = perf_counter()
        diagnostics = FrameDiagnostics(
            inference_ms=(inference_finished - started) * 1000.0,
            postprocess_ms=(finished - inference_finished) * 1000.0,
            total_ms=(finished - started) * 1000.0,
            lane_count=tracked_lane_count,
            road_area_ratio=float(np.count_nonzero(road_mask) / road_mask.size),
            used_road_fallback=self._road_filter.used_fallback,
            lane_confidence=lane_confidence,
            lane_rejected=lane_rejected,
        )
        return DetectionResult(
            road_mask=road_mask,
            lane_mask=lane_mask,
            overlay=overlay,
            diagnostics=diagnostics,
        )
