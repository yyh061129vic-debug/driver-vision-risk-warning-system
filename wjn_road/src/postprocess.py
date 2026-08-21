"""道路分割后处理：边缘修边、车底排除与道路线检测。"""

from __future__ import annotations

from typing import Any

import numpy as np

from driver_vision_risk.models.segformer import binary_inner_boundary


def _cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("postprocess requires opencv-python-headless") from exc
    return cv2


def _elliptical_kernel(size: int):
    cv2 = _cv2()
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _component_stats(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    cv2 = _cv2()
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return component_count, labels, stats


def refine_road_mask(
    road_mask: np.ndarray,
    confidence: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int | float]]:
    """通过连通域与形态学处理修整道路边缘。"""

    cv2 = _cv2()
    mask = road_mask.astype(np.bool_).copy()
    if mask.ndim != 2:
        raise ValueError("road mask must be 2D")

    height, width = mask.shape
    area = height * width
    refine_cfg = config.get("edge_refine", {}) if config.get("enabled", True) else {}
    min_component_ratio = float(refine_cfg.get("min_component_ratio", 0.0025))
    min_component_area = max(32, int(area * min_component_ratio))
    bottom_focus_ratio = float(refine_cfg.get("bottom_focus_ratio", 0.18))
    confidence_floor = float(refine_cfg.get("confidence_floor", 0.35))
    close_kernel = int(refine_cfg.get("close_kernel", 11))
    open_kernel = int(refine_cfg.get("open_kernel", 5))
    max_hole_area = int(refine_cfg.get("max_hole_area", max(256, area * 0.004)))

    confident_mask = mask & (confidence >= confidence_floor)
    refined = mask.astype(np.uint8) * 255
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _elliptical_kernel(close_kernel), iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, _elliptical_kernel(open_kernel), iterations=1)
    refined = (refined > 0) | confident_mask

    component_count, labels, stats = _component_stats(refined)
    keep = np.zeros_like(refined, dtype=np.bool_)
    bottom_anchor_y = int(height * (1.0 - bottom_focus_ratio))
    kept_components = 0
    for component_id in range(1, component_count):
        x, y, w, h, component_area = stats[component_id]
        if component_area < min_component_area:
            continue
        touches_bottom_focus = (y + h) >= bottom_anchor_y
        if touches_bottom_focus or component_area >= min_component_area * 3:
            keep |= labels == component_id
            kept_components += 1

    if not keep.any():
        keep = refined.copy()

    background = (~keep).astype(np.uint8)
    hole_count, hole_labels, hole_stats = _component_stats(background)
    filled_holes = 0
    for component_id in range(1, hole_count):
        x, y, w, h, component_area = hole_stats[component_id]
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border and component_area <= max_hole_area:
            keep[hole_labels == component_id] = True
            filled_holes += 1

    smoothed = cv2.GaussianBlur(keep.astype(np.uint8) * 255, (5, 5), sigmaX=0)
    refined_mask = smoothed >= 96
    refined_mask |= confident_mask
    return refined_mask.astype(np.bool_), {
        "kept_components": kept_components,
        "filled_holes": filled_holes,
        "removed_pixels": int(mask.sum() - np.logical_and(mask, refined_mask).sum()),
    }


def build_vehicle_exclusion_mask(
    class_map: np.ndarray,
    road_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """基于车辆语义区域构造车底排除掩码。"""

    vehicle_cfg = config.get("vehicle_exclusion", {})
    if not vehicle_cfg.get("enabled", True):
        return np.zeros_like(road_mask, dtype=np.bool_), []

    vehicle_class_ids = tuple(int(value) for value in vehicle_cfg.get("vehicle_class_ids", [11, 12, 13, 14, 15, 16, 17, 18]))
    min_area = int(vehicle_cfg.get("min_component_area", 160))
    min_width = int(vehicle_cfg.get("min_component_width", 12))
    min_height = int(vehicle_cfg.get("min_component_height", 12))
    start_ratio = float(vehicle_cfg.get("bottom_start_ratio", 0.55))
    expand_y_ratio = float(vehicle_cfg.get("bottom_expand_ratio", 0.28))
    expand_x_ratio = float(vehicle_cfg.get("side_expand_ratio", 0.12))
    min_expand_px = int(vehicle_cfg.get("min_expand_px", 10))

    vehicle_mask = np.isin(class_map, vehicle_class_ids)
    component_count, labels, stats = _component_stats(vehicle_mask)
    exclusion_mask = np.zeros_like(road_mask, dtype=np.bool_)
    boxes: list[dict[str, int]] = []
    height, width = road_mask.shape

    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        if area < min_area or w < min_width or h < min_height:
            continue
        start_y = min(height, y + max(1, int(h * start_ratio)))
        end_y = min(height, y + h + max(min_expand_px, int(h * expand_y_ratio)))
        if start_y >= end_y:
            continue
        pad_x = max(1, int(w * expand_x_ratio))
        left = max(0, x - pad_x)
        right = min(width, x + w + pad_x)
        exclusion_mask[start_y:end_y, left:right] = True
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "exclude_left": int(left),
                "exclude_top": int(start_y),
                "exclude_right": int(right),
                "exclude_bottom": int(end_y),
            }
        )

    exclusion_mask &= road_mask
    return exclusion_mask, boxes


def detect_lane_markings(
    rgb_image: np.ndarray,
    road_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, int | float | str]], dict[str, np.ndarray]]:
    """检测道路白线/黄线，并输出线段定位与基础分类。"""

    lane_cfg = config.get("lane_marking", {})
    if not lane_cfg.get("enabled", True):
        empty = np.zeros_like(road_mask, dtype=np.uint8)
        return empty, [], {"white": empty.copy(), "yellow": empty.copy()}

    cv2 = _cv2()
    height, width = road_mask.shape
    focus_top_ratio = float(lane_cfg.get("focus_top_ratio", 0.45))
    lower_focus = np.zeros_like(road_mask, dtype=np.uint8)
    lower_focus[int(height * focus_top_ratio) :, :] = 255
    constrained_mask = (road_mask.astype(np.uint8) * 255) & lower_focus

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    white_low = np.asarray(lane_cfg.get("white_hsv_low", [0, 0, 170]), dtype=np.uint8)
    white_high = np.asarray(lane_cfg.get("white_hsv_high", [180, 70, 255]), dtype=np.uint8)
    yellow_low = np.asarray(lane_cfg.get("yellow_hsv_low", [10, 60, 110]), dtype=np.uint8)
    yellow_high = np.asarray(lane_cfg.get("yellow_hsv_high", [45, 255, 255]), dtype=np.uint8)
    target_color = str(lane_cfg.get("target_color", "white")).lower()
    white_mask = cv2.inRange(hsv, white_low, white_high)
    yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
    contrast = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_TOPHAT, _elliptical_kernel(int(lane_cfg.get("contrast_kernel", 13))))
    edges = cv2.Canny(gray, int(lane_cfg.get("canny_low", 60)), int(lane_cfg.get("canny_high", 180)))
    color_candidate = (
        white_mask
        if target_color == "white"
        else yellow_mask
        if target_color == "yellow"
        else cv2.bitwise_or(white_mask, yellow_mask)
    )
    candidate = color_candidate
    candidate = cv2.bitwise_or(candidate, contrast)
    candidate = cv2.bitwise_and(candidate, constrained_mask)
    candidate = cv2.bitwise_and(candidate, edges | candidate)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _elliptical_kernel(int(lane_cfg.get("close_kernel", 5))), iterations=1)
    candidate = cv2.dilate(candidate, _elliptical_kernel(int(lane_cfg.get("dilate_kernel", 3))), iterations=1)
    lane_mask = candidate.copy()
    min_component_area = int(lane_cfg.get("min_component_area", 48))
    component_count, labels, stats = _component_stats(lane_mask > 0)
    filtered_lane_mask = np.zeros_like(lane_mask, dtype=np.uint8)
    for component_id in range(1, component_count):
        _, _, _, _, area = stats[component_id]
        if area < min_component_area:
            continue
        filtered_lane_mask[labels == component_id] = 255
    lane_mask = filtered_lane_mask

    raw_lines = cv2.HoughLinesP(
        candidate,
        1,
        np.pi / 180,
        threshold=int(lane_cfg.get("hough_threshold", 25)),
        minLineLength=int(lane_cfg.get("min_line_length", 28)),
        maxLineGap=int(lane_cfg.get("max_line_gap", 18)),
    )
    lane_candidates: list[dict[str, int | float | str]] = []
    if raw_lines is None:
        empty = np.zeros_like(lane_mask, dtype=np.uint8)
        return empty, lane_candidates, {"white": empty.copy(), "yellow": empty.copy()}

    line_rows = raw_lines[:, 0, :] if raw_lines.ndim == 3 else raw_lines
    center_x = width / 2.0
    min_line_length = float(lane_cfg.get("min_line_length", 28))
    min_support_ratio = float(lane_cfg.get("min_support_ratio", 0.03))
    max_support_ratio = float(lane_cfg.get("max_support_ratio", 1.0))
    max_support_ratio_white = float(lane_cfg.get("max_support_ratio_white", max_support_ratio))
    max_support_ratio_yellow = float(lane_cfg.get("max_support_ratio_yellow", max_support_ratio))
    dedupe_distance = float(lane_cfg.get("dedupe_distance_px", 28))
    dedupe_angle = float(lane_cfg.get("dedupe_angle_deg", 8.0))
    guidance_width = int(lane_cfg.get("guidance_width", 9))
    min_bottom_y = height * float(lane_cfg.get("min_bottom_y_ratio", 0.0))
    min_vertical_span = height * float(lane_cfg.get("min_vertical_span_ratio", 0.0))
    for line in line_rows:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        slope = 999.0 if dx == 0 else dy / float(dx)
        if abs(slope) < float(lane_cfg.get("min_abs_slope", 0.2)):
            continue
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < min_line_length:
            continue
        lower_y = max(y1, y2)
        vertical_span = abs(dy)
        if lower_y < min_bottom_y:
            continue
        if vertical_span < min_vertical_span:
            continue

        support_mask = np.zeros_like(candidate, dtype=np.uint8)
        cv2.line(support_mask, (x1, y1), (x2, y2), 255, int(lane_cfg.get("sample_width", 5)))
        support_pixels = support_mask > 0
        support_ratio = float(candidate[support_pixels].sum() / max(255.0, support_pixels.sum() * 255.0))
        if support_ratio < min_support_ratio:
            continue
        line_color = "white" if int(white_mask[support_pixels].sum()) >= int(yellow_mask[support_pixels].sum()) else "yellow"
        color_min_length = float(lane_cfg.get(f"min_line_length_{line_color}", min_line_length))
        color_min_vertical_span = height * float(lane_cfg.get(f"min_vertical_span_ratio_{line_color}", 0.0))
        if length < color_min_length:
            continue
        if vertical_span < color_min_vertical_span:
            continue
        if support_ratio > (max_support_ratio_white if line_color == "white" else max_support_ratio_yellow):
            continue
        if target_color in {"white", "yellow"} and line_color != target_color:
            continue
        style = "solid" if support_ratio >= float(lane_cfg.get("solid_support_ratio", 0.32)) else "dashed"
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0
        side = "left" if midpoint_x < center_x * 0.9 else "right" if midpoint_x > center_x * 1.1 else "center"
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        is_duplicate = False
        for existing in lane_candidates:
            existing_midpoint_x = (int(existing["x1"]) + int(existing["x2"])) / 2.0
            existing_midpoint_y = (int(existing["y1"]) + int(existing["y2"])) / 2.0
            midpoint_distance = ((existing_midpoint_x - midpoint_x) ** 2 + (existing_midpoint_y - midpoint_y) ** 2) ** 0.5
            if midpoint_distance <= dedupe_distance and abs(float(existing["angle_deg"]) - angle_deg) <= dedupe_angle:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        lane_candidates.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length_px": round(length, 2),
                "angle_deg": round(angle_deg, 2),
                "support_ratio": round(support_ratio, 4),
                "vertical_span_px": int(vertical_span),
                "bottom_y": int(lower_y),
                "color": line_color,
                "style": style,
                "side": side,
                "_score": round(length * support_ratio, 4),
            }
        )

    lane_candidates.sort(
        key=lambda item: (float(item["_score"]), float(item["length_px"])),
        reverse=True,
    )
    lane_lines = lane_candidates[: int(lane_cfg.get("max_lines", 64))]
    line_guidance = np.zeros_like(lane_mask, dtype=np.uint8)
    line_guidance_white = np.zeros_like(lane_mask, dtype=np.uint8)
    line_guidance_yellow = np.zeros_like(lane_mask, dtype=np.uint8)
    for line in lane_lines:
        x1 = int(line["x1"])
        y1 = int(line["y1"])
        x2 = int(line["x2"])
        y2 = int(line["y2"])
        cv2.line(line_guidance, (x1, y1), (x2, y2), 255, guidance_width)
        if line["color"] == "white":
            cv2.line(line_guidance_white, (x1, y1), (x2, y2), 255, guidance_width)
        else:
            cv2.line(line_guidance_yellow, (x1, y1), (x2, y2), 255, guidance_width)

    def keep_supported_components(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
        component_count, labels, _ = _component_stats(mask > 0)
        kept = np.zeros_like(mask, dtype=np.uint8)
        for component_id in range(1, component_count):
            component = labels == component_id
            if np.any(support[component] > 0):
                kept[component] = 255
        return kept

    def trim_to_guidance(mask: np.ndarray, support: np.ndarray, corridor_width: int) -> np.ndarray:
        if corridor_width <= 0:
            return mask
        corridor = cv2.dilate(support, _elliptical_kernel(corridor_width), iterations=1)
        return cv2.bitwise_and(mask, corridor)

    if not lane_lines:
        empty = np.zeros_like(lane_mask, dtype=np.uint8)
        return empty, lane_lines, {"white": empty.copy(), "yellow": empty.copy()}

    lane_mask = keep_supported_components(lane_mask, line_guidance)
    white_lane_mask = cv2.bitwise_and(lane_mask, white_mask)
    yellow_lane_mask = cv2.bitwise_and(lane_mask, yellow_mask)
    white_lane_mask = keep_supported_components(white_lane_mask, line_guidance_white)
    yellow_lane_mask = keep_supported_components(yellow_lane_mask, line_guidance_yellow)
    draw_width = int(lane_cfg.get("draw_width", 3))
    if draw_width > 1:
        expand_kernel = _elliptical_kernel(draw_width)
        white_lane_mask = cv2.dilate(white_lane_mask, expand_kernel, iterations=1)
        yellow_lane_mask = cv2.dilate(yellow_lane_mask, expand_kernel, iterations=1)
    white_lines = [line for line in lane_lines if line["color"] == "white"]
    white_total_length = sum(float(line["length_px"]) for line in white_lines)
    white_area_per_length = float(np.count_nonzero(white_lane_mask)) / max(1.0, white_total_length)
    white_shrink_ratio = float(lane_cfg.get("white_shrink_area_per_length", 0.0))
    white_shrink_max_count = int(lane_cfg.get("white_shrink_max_line_count", 0))
    white_shrink_corridor = int(lane_cfg.get("white_shrink_corridor_width", 0))
    if (
        white_shrink_ratio > 0.0
        and white_shrink_corridor > 0
        and white_total_length > 0.0
        and len(white_lines) <= max(0, white_shrink_max_count)
        and white_area_per_length > white_shrink_ratio
    ):
        white_lane_mask = trim_to_guidance(white_lane_mask, line_guidance_white, white_shrink_corridor)
    white_lane_mask = cv2.bitwise_and(white_lane_mask, constrained_mask)
    yellow_lane_mask = cv2.bitwise_and(yellow_lane_mask, constrained_mask)
    lane_mask = cv2.bitwise_or(white_lane_mask, yellow_lane_mask)
    for line in lane_lines:
        line.pop("_score", None)
    return lane_mask, lane_lines, {"white": white_lane_mask, "yellow": yellow_lane_mask}


def _joint_bilateral_filter(
    confidence: np.ndarray,
    rgb_image: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
) -> np.ndarray:
    """联合双边滤波：以 RGB 为引导，对二值/置信图做保边平滑。

    近似 DenseCRF 的 pairwise 双边项（颜色相近度 + 空间距离），使道路边界
    吸附到真实图像颜色边缘，同时抑制孤立毛刺。无 pydensecrf 依赖。
    """

    height, width = confidence.shape
    rgb = rgb_image.astype(np.float32) / 255.0
    conf = confidence.astype(np.float32)
    pad = int(radius)
    conf_pad = np.pad(conf, pad, mode="edge")
    rgb_pad = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    numerator = np.zeros((height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)
    spatial_denom = 2.0 * sigma_spatial * sigma_spatial
    color_denom = 2.0 * sigma_color * sigma_color
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_weight = float(np.exp(-(dx * dx + dy * dy) / spatial_denom))
            if spatial_weight < 1e-4:
                continue
            c_shift = conf_pad[pad + dy : pad + dy + height, pad + dx : pad + dx + width]
            r_shift = rgb_pad[pad + dy : pad + dy + height, pad + dx : pad + dx + width]
            diff_sq = ((r_shift - rgb) ** 2).sum(axis=2)
            color_weight = np.exp(-diff_sq / color_denom)
            weight = spatial_weight * color_weight
            numerator += weight * c_shift
            denominator += weight
    return numerator / np.maximum(denominator, 1e-6)


def refine_road_mask_crf(
    road_mask: np.ndarray,
    rgb_image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """方案 B：边缘感知精修（联合双边滤波近似 DenseCRF）。"""

    crf_cfg = config.get("crf", {})
    if not crf_cfg.get("enabled", True):
        return road_mask.astype(np.bool_), {"crf_applied": False}

    radius = int(crf_cfg.get("radius", 3))
    sigma_spatial = float(crf_cfg.get("sigma_spatial", 3.0))
    sigma_color = float(crf_cfg.get("sigma_color", 0.10))
    threshold = float(crf_cfg.get("threshold", 0.5))
    indicator = road_mask.astype(np.float32)
    soft = _joint_bilateral_filter(indicator, rgb_image, radius, sigma_spatial, sigma_color)
    refined = soft >= threshold
    return refined.astype(np.bool_), {"crf_applied": True, "crf_radius": radius}


def refine_road_mask_snake(
    road_mask: np.ndarray,
    rgb_image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """方案 C：主动轮廓式边缘精修（最大外轮廓梯度吸附 + 圆滑回填）。"""

    from scipy.ndimage import gaussian_filter1d

    cv2 = _cv2()
    snake_cfg = config.get("snake", {})
    if not snake_cfg.get("enabled", True):
        return road_mask.astype(np.bool_), {"snake_applied": False}

    mask_u8 = (road_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return road_mask.astype(np.bool_), {"snake_applied": False}
    contour = max(contours, key=cv2.contourArea)
    min_area = float(snake_cfg.get("min_contour_area", 5000))
    if cv2.contourArea(contour) < min_area:
        return road_mask.astype(np.bool_), {"snake_applied": False}

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    grad_mag = cv2.GaussianBlur(grad_mag, (5, 5), 0)

    points = contour.reshape(-1, 2).astype(np.int32)
    target_count = int(snake_cfg.get("target_points", 400))
    if len(points) > target_count:
        step = max(1, len(points) // target_count)
        points = points[::step]
    snap_radius = int(snake_cfg.get("snap_radius", 4))
    for _ in range(int(snake_cfg.get("snap_iterations", 1))):
        snapped = points.copy()
        for index, (x, y) in enumerate(points):
            y0 = max(0, y - snap_radius)
            y1 = min(gray.shape[0], y + snap_radius + 1)
            x0 = max(0, x - snap_radius)
            x1 = min(gray.shape[1], x + snap_radius + 1)
            window = grad_mag[y0:y1, x0:x1]
            ly, lx = np.unravel_index(int(np.argmax(window)), window.shape)
            snapped[index] = [x0 + lx, y0 + ly]
        points = snapped

    sigma = float(snake_cfg.get("smooth_sigma", 2.0))
    xs = gaussian_filter1d(points[:, 0].astype(np.float32), sigma=sigma, mode="wrap")
    ys = gaussian_filter1d(points[:, 1].astype(np.float32), sigma=sigma, mode="wrap")
    smoothed = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    refined = np.zeros_like(mask_u8)
    cv2.fillPoly(refined, [smoothed], 255)
    return (refined > 0).astype(np.bool_), {"snake_applied": True, "snake_points": int(len(points))}


def trim_road_edge_by_lanes(
    road_mask: np.ndarray,
    confidence: np.ndarray,
    lane_lines: list[dict[str, int | float | str]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """方案 A：车道线引导切边。

    用最外侧左/右车道线作为硬约束，仅剔除底部区域中、位于最外侧车道线之外且
    低置信的道路像素；无车道线时不做任何改动（应对雪盖等无标线路段）。
    """

    trim_cfg = config.get("lane_trim", {})
    enabled = trim_cfg.get("enabled", True)
    if not enabled or not lane_lines:
        return road_mask.astype(np.bool_), {"lane_trim_applied": enabled and bool(lane_lines)}

    height, width = road_mask.shape
    left_lines = [line for line in lane_lines if line.get("side") == "left"]
    right_lines = [line for line in lane_lines if line.get("side") == "right"]
    if not left_lines and not right_lines:
        return road_mask.astype(np.bool_), {"lane_trim_applied": True}

    bottom_ratio = float(trim_cfg.get("bottom_ratio", 0.6))
    margin = int(trim_cfg.get("margin", 8))
    confidence_floor = float(trim_cfg.get("confidence_floor", 0.55))
    top_y = int(height * (1.0 - bottom_ratio))
    trim_mask = np.zeros_like(road_mask, dtype=np.bool_)
    bottom_band = np.zeros_like(road_mask, dtype=np.bool_)
    bottom_band[top_y:, :] = True
    low_confidence = confidence < confidence_floor

    if left_lines:
        left_x = min(min(int(line["x1"]), int(line["x2"])) for line in left_lines)
        outside_left = np.zeros_like(road_mask, dtype=np.bool_)
        if left_x - margin > 0:
            outside_left[:, : left_x - margin] = True
        trim_mask |= bottom_band & outside_left & low_confidence & road_mask

    if right_lines:
        right_x = max(max(int(line["x1"]), int(line["x2"])) for line in right_lines)
        outside_right = np.zeros_like(road_mask, dtype=np.bool_)
        if right_x + margin < width:
            outside_right[:, right_x + margin :] = True
        trim_mask |= bottom_band & outside_right & low_confidence & road_mask

    return (road_mask & ~trim_mask).astype(np.bool_), {
        "lane_trim_applied": True,
        "lane_trim_pixels": int(trim_mask.sum()),
    }


def apply_edge_cut(
    road_mask: np.ndarray,
    confidence: np.ndarray,
    rgb_image: np.ndarray,
    lane_lines: list[dict[str, int | float | str]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """按配置依次应用 B(CRF)/C(Snake)/A(车道线切边)。"""

    edge_cfg = config.get("edge_cut", {})
    result = road_mask.astype(np.bool_)
    stats: dict[str, int | float | bool] = {}
    if edge_cfg.get("crf_enabled", False):
        result, crf_stats = refine_road_mask_crf(result, rgb_image, config)
        stats.update(crf_stats)
    if edge_cfg.get("snake_enabled", False):
        result, snake_stats = refine_road_mask_snake(result, rgb_image, config)
        stats.update(snake_stats)
    if edge_cfg.get("lane_trim_enabled", False):
        result, trim_stats = trim_road_edge_by_lanes(result, confidence, lane_lines, config)
        stats.update(trim_stats)
    return result, stats


def enhance_drivable_prediction(
    rgb_image: np.ndarray,
    road_mask: np.ndarray,
    confidence: np.ndarray,
    class_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """汇总主流程增强结果。"""

    postprocess_cfg = config.get("postprocess", {})
    refined_mask, refine_stats = refine_road_mask(road_mask, confidence, postprocess_cfg)
    vehicle_exclusion_mask, vehicle_boxes = build_vehicle_exclusion_mask(
        class_map,
        refined_mask,
        postprocess_cfg,
    )
    refined_without_vehicle = refined_mask & ~vehicle_exclusion_mask

    # Keep lane detection independent from underbody exclusion so the drivable
    # mask fix does not suppress nearby lane-marking context.
    lane_mask, lane_lines, lane_masks_by_color = detect_lane_markings(
        rgb_image,
        refined_mask,
        postprocess_cfg,
    )

    edge_cut_mask, edge_cut_stats = apply_edge_cut(
        refined_without_vehicle,
        confidence,
        rgb_image,
        lane_lines,
        postprocess_cfg,
    )
    boundary = binary_inner_boundary(
        edge_cut_mask.astype(np.bool_),
        width=int(config["visualization"]["boundary_width"]),
    )
    return {
        "raw_mask": road_mask.astype(np.bool_),
        "refined_mask": refined_without_vehicle.astype(np.bool_),
        "edge_cut_mask": edge_cut_mask.astype(np.bool_),
        "boundary": boundary.astype(np.bool_),
        "vehicle_exclusion_mask": vehicle_exclusion_mask.astype(np.bool_),
        "vehicle_boxes": vehicle_boxes,
        "lane_mask": lane_mask.astype(np.uint8),
        "lane_mask_white": lane_masks_by_color["white"].astype(np.uint8),
        "lane_mask_yellow": lane_masks_by_color["yellow"].astype(np.uint8),
        "lane_lines": lane_lines,
        "stats": {
            **refine_stats,
            **edge_cut_stats,
            "vehicle_count": len(vehicle_boxes),
            "excluded_pixels": int(vehicle_exclusion_mask.sum()),
            "lane_pixels": int((lane_mask > 0).sum()),
            "lane_pixels_white": int((lane_masks_by_color["white"] > 0).sum()),
            "lane_pixels_yellow": int((lane_masks_by_color["yellow"] > 0).sum()),
            "lane_line_count_white": sum(1 for item in lane_lines if item["color"] == "white"),
            "lane_line_count_yellow": sum(1 for item in lane_lines if item["color"] == "yellow"),
            "lane_line_count": len(lane_lines),
        },
    }
