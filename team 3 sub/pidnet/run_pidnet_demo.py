"""team 3 sub/pidnet 的独立图片与视频推理脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "pidnet_s_cityscapes.yaml"


@dataclass(frozen=True)
class DrivableAreaPrediction:
    mask: np.ndarray
    boundary: np.ndarray
    confidence: np.ndarray
    class_map: np.ndarray
    latency_ms: float
    device: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_inner_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if width < 1:
        raise ValueError("boundary width must be at least one pixel")
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


class PidnetDrivableAreaSegmenter:
    @staticmethod
    def preload_windows_cuda_dlls() -> None:
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
                f"model directory not found: {self.model_directory}; run download_pidnet_model.py first"
            )

        self.preload_windows_cuda_dlls()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("missing onnxruntime dependency") from exc
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
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.device = f"onnx:{self.session.get_providers()[0]}"

    def resize_logits(self, logits: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless is required") from exc
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
        logits = self.resize_logits(logits, target_height, target_width)
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


def render_overlay(
    image: Image.Image,
    prediction: DrivableAreaPrediction,
    road_color: tuple[int, int, int],
    boundary_color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    color = np.asarray(road_color, dtype=np.float32)
    base[prediction.mask] = base[prediction.mask] * (1.0 - alpha) + color * alpha
    base[prediction.boundary] = np.asarray(boundary_color, dtype=np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def prediction_metrics(prediction: DrivableAreaPrediction) -> dict[str, float | int | str]:
    road_pixels = int(prediction.mask.sum())
    pixel_count = int(prediction.mask.size)
    mean_confidence = float(prediction.confidence[prediction.mask].mean()) if road_pixels else 0.0
    return {
        "device": prediction.device,
        "latency_ms": round(prediction.latency_ms, 3),
        "pixel_count": pixel_count,
        "road_pixel_count": road_pixels,
        "road_coverage": round(road_pixels / pixel_count, 6),
        "mean_road_confidence": round(mean_confidence, 6),
    }


def base_metadata(
    input_path: Path,
    config: dict[str, Any],
    prediction_metrics_data: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    try:
        recorded_input_path = input_path.relative_to(project_root).as_posix()
    except ValueError:
        recorded_input_path = input_path.name

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "input": {
            "path": recorded_input_path,
            "sha256": sha256(input_path),
        },
        "model": {
            "id": config["model"]["id"],
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "weights_sha256": config["model"]["weights_sha256"],
            "road_class_ids": config["segmentation"]["road_class_ids"],
            "road_class_names": config["segmentation"]["road_class_names"],
        },
        "runtime": {
            "python": platform.python_version(),
            "framework": config["runtime"]["framework"],
            "onnxruntime": __import__("onnxruntime").__version__,
            **prediction_metrics_data,
        },
    }


def run_image_demo(input_path: Path, output_directory: Path, segmenter: PidnetDrivableAreaSegmenter) -> dict[str, Any]:
    image = Image.open(input_path).convert("RGB")
    prediction = segmenter.predict(image)
    config = segmenter.config
    visualization = config["visualization"]
    overlay = render_overlay(
        image,
        prediction,
        tuple(visualization["road_color_rgb"]),
        tuple(visualization["boundary_color_rgb"]),
        float(visualization["overlay_alpha"]),
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "drivable-mask.png"
    boundary_path = output_directory / "drivable-boundary.png"
    confidence_path = output_directory / "road-confidence.png"
    overlay_path = output_directory / "overlay.png"
    Image.fromarray(prediction.mask.astype(np.uint8) * 255).save(mask_path)
    Image.fromarray(prediction.boundary.astype(np.uint8) * 255).save(boundary_path)
    Image.fromarray(np.clip(prediction.confidence * 255.0, 0, 255).astype(np.uint8)).save(confidence_path)
    overlay.save(overlay_path, optimize=True)

    metrics = prediction_metrics(prediction)
    metadata = base_metadata(input_path, config, metrics, segmenter.project_root)
    metadata["input"].update({"width": image.width, "height": image.height, "type": "image"})
    metadata["outputs"] = {
        "mask": mask_path.name,
        "boundary": boundary_path.name,
        "confidence": confidence_path.name,
        "overlay": overlay_path.name,
    }
    for name, filename in list(metadata["outputs"].items()):
        metadata["outputs"][f"{name}_sha256"] = sha256(output_directory / filename)
    (output_directory / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def run_video_demo(input_path: Path, output_directory: Path, segmenter: PidnetDrivableAreaSegmenter) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video input requires opencv-python-headless") from exc

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "overlay.mp4"
    codec = cv2.VideoWriter_fourcc(*str(segmenter.config["video"]["output_codec"]))
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot create output video with configured codec")

    frame_metrics: list[dict[str, Any]] = []
    visualization = segmenter.config["visualization"]
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            prediction = segmenter.predict(image)
            overlay = render_overlay(
                image,
                prediction,
                tuple(visualization["road_color_rgb"]),
                tuple(visualization["boundary_color_rgb"]),
                float(visualization["overlay_alpha"]),
            )
            writer.write(cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR))
            frame_metrics.append(prediction_metrics(prediction))
    finally:
        capture.release()
        writer.release()
    if not frame_metrics:
        raise RuntimeError("video contains no readable frames")

    aggregate = {
        "device": frame_metrics[0]["device"],
        "frame_count": len(frame_metrics),
        "average_latency_ms": round(
            sum(float(item["latency_ms"]) for item in frame_metrics) / len(frame_metrics), 3
        ),
        "average_road_coverage": round(
            sum(float(item["road_coverage"]) for item in frame_metrics) / len(frame_metrics), 6
        ),
    }
    metadata = base_metadata(input_path, segmenter.config, aggregate, segmenter.project_root)
    metadata["input"].update({"width": width, "height": height, "fps": fps, "type": "video"})
    metadata["outputs"] = {"overlay": output_path.name, "overlay_sha256": sha256(output_path)}
    (output_directory / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="input image or video path")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="PIDNet yaml config path")
    args = parser.parse_args()

    input_path = args.input.resolve()
    config_path = args.config.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    segmenter = PidnetDrivableAreaSegmenter(config_path, ROOT)
    suffix = input_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        result = run_image_demo(input_path, args.output.resolve(), segmenter)
    elif suffix in VIDEO_SUFFIXES:
        result = run_video_demo(input_path, args.output.resolve(), segmenter)
    else:
        raise ValueError(f"unsupported input extension: {suffix}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
