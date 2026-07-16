"""固定版本 SegFormer 的可行驶区域推理封装。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


@dataclass(frozen=True)
class DrivableAreaPrediction:
    """单帧道路分割结果；所有像素数组均保持输入图像的原始分辨率。"""

    mask: np.ndarray
    boundary: np.ndarray
    confidence: np.ndarray
    class_map: np.ndarray
    latency_ms: float
    device: str


def binary_inner_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """计算二值掩码的内边界，不依赖 OpenCV 或 SciPy。"""

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if width < 1:
        raise ValueError("boundary width must be at least one pixel")
    # 每轮使用四邻域腐蚀一圈，再用原掩码减去腐蚀结果得到内边界。
    eroded = mask.copy()
    for _ in range(width):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return mask & ~eroded


class SegformerDrivableAreaSegmenter:
    """加载本地固定权重，并预测 Cityscapes 中的 ``road`` 类别。"""

    def __init__(self, config_path: Path, project_root: Path) -> None:
        """读取模型配置、校验运行设备并初始化图像处理器和分割模型。"""

        self.config_path = config_path
        self.project_root = project_root
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = self.config["model"]
        runtime_config = self.config["runtime"]
        self.model_directory = project_root / model_config["local_directory"]
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"model directory not found: {self.model_directory}; run scripts/download_segmentation_model.py"
            )

        # 推理依赖按需导入；只运行元数据工具时不强制加载大型框架。
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "segmentation dependencies are missing; install the inference extra"
            ) from exc

        self.torch = torch
        requested_device = str(runtime_config["device"])
        # ``auto`` 仅在 PyTorch 确认 CUDA 可用时选择 GPU。
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        self.device = torch.device(requested_device)
        if self.device.type == "cpu":
            torch.set_num_threads(int(runtime_config.get("torch_threads", 1)))

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_directory,
            local_files_only=True,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_directory,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.road_class_ids = tuple(int(value) for value in self.config["segmentation"]["road_class_ids"])
        self.confidence_threshold = self.config["segmentation"].get("confidence_threshold")
        self.boundary_width = int(self.config["visualization"]["boundary_width"])

        # 以权重自带标签表为准，防止配置误把其他类别当作道路。
        id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        expected_names = tuple(self.config["segmentation"]["road_class_names"])
        actual_names = tuple(id2label[class_id] for class_id in self.road_class_ids)
        if actual_names != expected_names:
            raise RuntimeError(
                f"road class mismatch: configured {expected_names}, checkpoint reports {actual_names}"
            )

    def predict(self, image: Image.Image) -> DrivableAreaPrediction:
        """对一帧 RGB 图像推理并返回道路掩码、边界、置信度和类别图。"""

        torch = self.torch
        rgb_image = image.convert("RGB")
        target_size = (rgb_image.height, rgb_image.width)
        # CUDA 默认异步执行；计时前后同步，避免把尚未完成的 GPU 工作漏掉。
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # 模型输出分辨率低于输入，需要先恢复到原图尺寸再逐像素判类。
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)
            class_map_tensor = logits.argmax(dim=1)[0]
            road_probabilities = probabilities[0, list(self.road_class_ids)].amax(dim=0)
            # 使用配置中的道路类别集合构造二值掩码，便于以后扩展多个可行驶类。
            road_mask = torch.zeros_like(class_map_tensor, dtype=torch.bool)
            for class_id in self.road_class_ids:
                road_mask |= class_map_tensor == class_id
            # 当前基线为 null，即只采用 argmax；设置阈值时才做额外过滤。
            if self.confidence_threshold is not None:
                road_mask &= road_probabilities >= float(self.confidence_threshold)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0

        # 输出统一转成 NumPy，供图像和视频入口复用，不泄漏框架张量。
        mask = road_mask.cpu().numpy().astype(np.bool_)
        confidence = road_probabilities.cpu().numpy().astype(np.float32)
        class_map = class_map_tensor.cpu().numpy().astype(np.uint8)
        boundary = binary_inner_boundary(mask, width=self.boundary_width)
        return DrivableAreaPrediction(
            mask=mask,
            boundary=boundary,
            confidence=confidence,
            class_map=class_map,
            latency_ms=latency_ms,
            device=str(self.device),
        )
