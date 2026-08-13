"""PIDNet-S ONNX 版本的可行驶区域推理封装。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from driver_vision_risk.models.segformer import DrivableAreaPrediction, binary_inner_boundary


class PidnetDrivableAreaSegmenter:
    """加载本地 PIDNet-S ONNX 资产，并预测 Cityscapes 中的 ``road`` 类别。"""

    @staticmethod
    def _preload_windows_cuda_dlls() -> None:
        """在 Windows 上预加载 PyTorch 自带的 CUDA DLL，供 ONNX Runtime 复用。"""

        if os.name != "nt":
            return
        try:
            import torch
        except ImportError:
            return
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))

    def __init__(self, config_path: Path, project_root: Path) -> None:
        self.config_path = config_path
        self.project_root = project_root
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = self.config["model"]
        runtime_config = self.config["runtime"]
        self.model_directory = project_root / model_config["local_directory"]
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"model directory not found: {self.model_directory}; run scripts/download_pidnet_model.py"
            )

        self._preload_windows_cuda_dlls()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "PIDNet dependencies are missing; install onnxruntime or the inference extra"
            ) from exc
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        self.ort = ort
        self.model_path = self.model_directory / model_config["weights_file"]
        self.external_data_path = self.model_directory / model_config["external_data_file"]
        self.metadata_path = self.model_directory / model_config["metadata_file"]
        self.labels_path = self.model_directory / model_config["labels_file"]
        for required in (
            self.model_path,
            self.external_data_path,
            self.metadata_path,
            self.labels_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(f"missing PIDNet asset: {required}")

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        model_files = metadata.get("model_files", {}).get(self.model_path.name, {})
        input_spec = model_files.get("inputs", {}).get("image", {})
        output_spec = model_files.get("outputs", {}).get("mask", {})
        self.input_height = int(runtime_config.get("input_height", input_spec["shape"][2]))
        self.input_width = int(runtime_config.get("input_width", input_spec["shape"][3]))
        if tuple(input_spec.get("shape", [])) != (1, 3, self.input_height, self.input_width):
            raise RuntimeError("PIDNet input shape does not match configured runtime size")
        if tuple(output_spec.get("shape", []))[1] != 19:
            raise RuntimeError("PIDNet exported asset must output 19 Cityscapes classes")

        self.input_name = "image"
        self.output_name = "mask"
        self.road_class_ids = tuple(int(value) for value in self.config["segmentation"]["road_class_ids"])
        self.confidence_threshold = self.config["segmentation"].get("confidence_threshold")
        self.boundary_width = int(self.config["visualization"]["boundary_width"])
        expected_names = tuple(self.config["segmentation"]["road_class_names"])
        labels = tuple(self.labels_path.read_text(encoding="utf-8").splitlines())
        actual_names = tuple(labels[class_id] for class_id in self.road_class_ids)
        if actual_names != expected_names:
            raise RuntimeError(
                f"road class mismatch: configured {expected_names}, checkpoint reports {actual_names}"
            )

        requested_device = str(runtime_config.get("device", "auto"))
        available = list(ort.get_available_providers())
        if requested_device == "cpu":
            desired = ["CPUExecutionProvider"]
        else:
            desired = list(runtime_config.get("provider_priority", ["CPUExecutionProvider"]))
        providers = [provider for provider in desired if provider in available]
        if not providers:
            raise RuntimeError(f"no requested onnxruntime provider is available: {desired}")
        self.providers = providers
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        self.device = f"onnx:{self.session.get_providers()[0]}"

    def _resize_logits(self, logits: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("PIDNet resizing requires opencv-python-headless") from exc
        resized = np.empty((logits.shape[0], target_height, target_width), dtype=np.float32)
        for class_index in range(logits.shape[0]):
            resized[class_index] = cv2.resize(
                logits[class_index],
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return resized

    def predict(self, image: Image.Image) -> DrivableAreaPrediction:
        rgb_image = image.convert("RGB")
        target_height, target_width = rgb_image.height, rgb_image.width
        network_input = rgb_image.resize((self.input_width, self.input_height), Image.BILINEAR)
        array = np.asarray(network_input, dtype=np.float32) / 255.0
        nchw = np.transpose(array, (2, 0, 1))[None, ...]

        started = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: nchw})
        latency_ms = (time.perf_counter() - started) * 1000.0

        logits = np.asarray(outputs[0][0], dtype=np.float32)
        logits = self._resize_logits(logits, target_height, target_width)
        shifted = logits - logits.max(axis=0, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        class_map = logits.argmax(axis=0).astype(np.uint8)
        road_probabilities = probabilities[list(self.road_class_ids)].max(axis=0).astype(np.float32)
        road_mask = np.zeros_like(class_map, dtype=np.bool_)
        for class_id in self.road_class_ids:
            road_mask |= class_map == class_id
        if self.confidence_threshold is not None:
            road_mask &= road_probabilities >= float(self.confidence_threshold)
        boundary = binary_inner_boundary(road_mask, width=self.boundary_width)
        return DrivableAreaPrediction(
            mask=road_mask,
            boundary=boundary,
            confidence=road_probabilities,
            class_map=class_map,
            latency_ms=latency_ms,
            device=self.device,
        )
