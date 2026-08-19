import os
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np
import torch
from PIL import Image

from config.config import Config
from predict import (
    load_segformer,
    predict_road,
)
from utils.utils import ensure_dir

# 覆盖色（BGR，视频帧为 BGR）：翠绿
OVERLAY_COLOR_BGR = np.array(
    (113, 204, 46),
    dtype=np.float32,
)

# 车辆框颜色（BGR）：红
BOX_COLOR_BGR = (0, 0, 255)


class JobCancelled(Exception):
    """任务被用户取消。"""


class InferenceService:
    """Web 推理服务：启动时加载模型，逐帧调用现有 predict 代码处理视频。"""

    def __init__(self):
        self.cfg = Config

        ensure_dir(self.cfg.MASK_OUTPUT_DIR)
        ensure_dir(self.cfg.RESULT_DIR)

        self.processor = None
        self.model = None
        self.road_class = 1
        self.device = self.cfg.DEVICE
        self.model_source = None
        self.yolo = None
        self.error = None

        self._jobs = {}
        self._job_lock = threading.Lock()

        # 检测 NVIDIA GPU：有则优先使用 NVDEC/NVENC 硬解码/硬编码，否则 CPU
        self.use_gpu = torch.cuda.is_available()

        self.nvenc_available = (
            self.use_gpu and self._check_nvenc()
        )

        self.nvdec_available = (
            self.use_gpu and self._check_nvdec()
        )

        try:
            (
                self.processor,
                self.model,
                self.road_class,
                self.device,
                self.model_source,
            ) = load_segformer(self.cfg)

            self.yolo = self._load_yolo()
        except Exception as exc:  # noqa: BLE001
            self.error = f"模型加载失败: {exc}"

    @property
    def ready(self):
        return self.error is None and self.model is not None

    # ---------------------------------------------------------------
    # 任务管理：POST 后立即返回 job_id，后台线程处理，前端轮询进度
    # ---------------------------------------------------------------

    def start_video_job(self, path: str, mode: str) -> str:
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")

        job_id = uuid.uuid4().hex[:16]

        with self._job_lock:
            self._jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "result": None,
                "error": None,
                "cancel_requested": False,
            }

        threading.Thread(
            target=self._run_job,
            args=(job_id, path, mode),
            daemon=True,
        ).start()

        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """请求取消任务；正在处理的帧循环会在下一帧停止并清理文件。"""
        with self._job_lock:
            job = self._jobs.get(job_id)

            if job is None:
                return False

            job["cancel_requested"] = True

        return True

    def _run_job(self, job_id: str, path: str, mode: str):
        def on_progress(processed: int, total: int):
            with self._job_lock:
                job = self._jobs[job_id]
                job["progress"] = (
                    processed / total if total else 0.0
                )
                job["processed"] = processed
                job["total"] = total

        def is_cancelled():
            with self._job_lock:
                return self._jobs[job_id].get(
                    "cancel_requested",
                    False,
                )

        try:
            result = self.process_video(
                path,
                mode,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )

            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "done"
                job["progress"] = 1.0
                job["result"] = result
        except JobCancelled:
            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "cancelled"
                job["error"] = None
        except Exception as exc:  # noqa: BLE001
            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "error"
                job["error"] = str(exc)

    def get_job(self, job_id: str):
        with self._job_lock:
            job = self._jobs.get(job_id)

        if job is None:
            return None

        return dict(job)

    # ---------------------------------------------------------------
    # 单帧推理（复用现有 predict.py / predict_fusion_yolo.py 逻辑）
    # ---------------------------------------------------------------

    def _seg_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        image = Image.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )

        return predict_road(
            self.processor,
            self.model,
            image,
            self.device,
            self.road_class,
        )

    def _load_yolo(self):
        """加载 YOLO 检测器；权重缺失时返回 None。"""
        if not os.path.exists(self.cfg.YOLO_WEIGHT):
            print(
                "YOLO 权重不存在:",
                self.cfg.YOLO_WEIGHT,
            )
            return None

        from ultralytics import YOLO

        return YOLO(self.cfg.YOLO_WEIGHT)

    def _detect_vehicles(self, frame_bgr):
        """YOLO 检测车辆，返回检测框列表（与 predict_fusion_yolo 语义一致）。"""
        if self.yolo is None:
            return []

        results = self.yolo(
            frame_bgr,
            verbose=False,
            device=self.device,
        )

        detections = []

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if (
                    cls in self.cfg.YOLO_VEHICLE_CLASSES
                    and conf > 0.3
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0],
                    )

                    detections.append({
                        "cls": cls,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                    })

        return detections

    def _fusion(self, frame_bgr: np.ndarray):
        """Fusion：返回 (融合后 road mask, 车辆检测框列表)。"""
        road_mask = self._seg_mask(frame_bgr)

        if self.yolo is None:
            return road_mask, []

        detections = self._detect_vehicles(frame_bgr)

        vehicle_mask = np.zeros(
            frame_bgr.shape[:2],
            dtype=np.uint8,
        )

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            vehicle_mask[y1:y2, x1:x2] = 1

        # 融合：车辆区域从可行驶道路中剔除
        fused = road_mask.copy()
        fused[vehicle_mask == 1] = 0

        return fused, detections

    def _seg_masks(self, frames_bgr):
        """批量 SegFormer 推理：一次将多帧送入 GPU，返回每帧 0/1 road mask。"""
        images = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames_bgr
        ]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values
            )

        height, width = frames_bgr[0].shape[:2]
        logits = outputs.logits  # [B, 150, h, w]
        batch_n = len(frames_bgr)

        # 避免 upsample 输出 tensor 过大：batch*150*H*W < 2^29（约 2GB）时
        # 批量插值（快）；否则逐帧插值，防止大图/大 batch 产生超大显存中间张量
        # （SegFormer 推理本身仍保持 batch，只有插值逐帧）。
        if batch_n * logits.shape[1] * height * width < 2**29:
            logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            return [
                (pred[i] == self.road_class).astype(np.uint8)
                for i in range(batch_n)
            ]

        masks = []

        for i in range(batch_n):
            logit = torch.nn.functional.interpolate(
                logits[i:i + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            pred = torch.argmax(
                logit,
                dim=1,
            ).cpu().numpy()

            masks.append(
                (pred[0] == self.road_class).astype(np.uint8)
            )

        return masks

    def _fusion_batch(self, frames_bgr):
        """批量 Fusion：返回 (每帧融合后 mask, 每帧检测框列表)。"""
        road_masks = self._seg_masks(frames_bgr)

        if self.yolo is None:
            return road_masks, [
                [] for _ in frames_bgr
            ]

        results = self.yolo(
            frames_bgr,
            verbose=False,
            device=self.device,
        )

        fused_masks = []
        detections_list = []

        for i, frame in enumerate(frames_bgr):

            detections = []

            for box in results[i].boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if (
                    cls in self.cfg.YOLO_VEHICLE_CLASSES
                    and conf > 0.3
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0],
                    )

                    detections.append({
                        "cls": cls,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                    })

            vehicle_mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8,
            )

            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                vehicle_mask[y1:y2, x1:x2] = 1

            fused = road_masks[i].copy()
            fused[vehicle_mask == 1] = 0

            fused_masks.append(fused)
            detections_list.append(detections)

        return fused_masks, detections_list

    def _process_frames(self, frames_bgr, mode):
        """对一批帧执行推理，返回 (输出帧列表, 含车辆帧数)。"""
        if mode == "segmentation":
            masks = self._seg_masks(frames_bgr)
            return [
                self._apply_overlay(frame, mask)
                for frame, mask in zip(frames_bgr, masks)
            ], 0

        fused_masks, detections_list = self._fusion_batch(frames_bgr)

        out_frames = []
        vehicle_frames = 0

        for frame, mask, detections in zip(
            frames_bgr,
            fused_masks,
            detections_list,
        ):
            out = self._apply_overlay(frame, mask)
            out = self._draw_boxes(out, detections)

            if detections:
                vehicle_frames += 1

            out_frames.append(out)

        return out_frames, vehicle_frames

    def _draw_boxes(
        self,
        frame_bgr: np.ndarray,
        detections,
    ) -> np.ndarray:
        """在帧上绘制车辆检测框（红色矩形 + 类别/置信度标签）。"""
        if not detections:
            return frame_bgr

        names = (
            getattr(self.yolo, "names", {})
            if self.yolo is not None
            else {}
        )

        out = frame_bgr.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            cls = detection["cls"]
            conf = detection["conf"]

            cv2.rectangle(
                out,
                (x1, y1),
                (x2, y2),
                BOX_COLOR_BGR,
                2,
            )

            label = f"{names.get(cls, str(cls))} {conf:.2f}"

            cv2.putText(
                out,
                label,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BOX_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )

        return out

    def _apply_overlay(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        frame = frame_bgr.astype(np.float32)

        blended = (
            OVERLAY_COLOR_BGR * 0.5 + frame * 0.5
        )

        out = frame.copy()
        out[mask == 1] = blended[mask == 1]

        return out.astype(np.uint8)

    # ---------------------------------------------------------------
    # 视频处理
    # ---------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        mode: str,
        on_progress=None,
        is_cancelled=None,
    ):
        """处理已落盘的上传视频（main 已流式保存，大小无限制）。"""
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")

        if mode not in ("segmentation", "fusion"):
            raise ValueError("mode 必须为 segmentation 或 fusion")

        token = uuid.uuid4().hex[:12]

        prepared_path = os.path.join(
            self.cfg.RESULT_DIR,
            f"input_{token}.mp4",
        )

        output_path = os.path.join(
            self.cfg.RESULT_DIR,
            f"output_{token}.mp4",
        )

        try:
            # 输入视频统一转码为 H.264（NVENC）+ faststart，保证浏览器可播放
            self._prepare_input(input_path, prepared_path)

            return self._run_video_pipeline(
                prepared_path,
                output_path,
                mode,
                on_progress,
                is_cancelled,
            )
        except Exception:
            for p in (prepared_path, output_path):
                if os.path.exists(p):
                    os.remove(p)
            raise

    def _process_image_bgr(self, image: np.ndarray, mode: str):
        """单张图片推理核心：返回 (输出图 BGR, 检测框列表)。"""
        if mode == "segmentation":
            mask = self._seg_mask(image)
            out = self._apply_overlay(image, mask)
            detections = []
        else:
            mask, detections = self._fusion(image)
            out = self._apply_overlay(image, mask)
            out = self._draw_boxes(out, detections)

        return out, detections

    def _image_result(self, image, out, detections, mode, token):
        """保存结果图并组装返回结构（JPG 格式，保存速度快且体积小）。"""
        input_path = os.path.join(
            self.cfg.RESULT_DIR,
            f"img_{token}_orig.jpg",
        )

        output_path = os.path.join(
            self.cfg.RESULT_DIR,
            f"img_{token}_result.jpg",
        )

        cv2.imwrite(input_path, image)
        cv2.imwrite(output_path, out)

        height, width = image.shape[:2]

        return {
            "mode": mode,
            "files": {
                "input": os.path.basename(input_path),
                "output": os.path.basename(output_path),
            },
            "metrics": {
                "resolution": f"{width}x{height}",
                "device": self.device,
                "vehicle_count": len(detections),
            },
            "model_source": self.model_source,
            "yolo": {
                "available": self.yolo is not None,
                "message": "ok" if self.yolo is not None
                else "YOLO model unavailable: 权重缺失",
            },
        }

    def process_image(self, data: bytes, mode: str):
        """单张图片推理（同步执行），返回结果图路径与基本指标。"""
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")

        if mode not in ("segmentation", "fusion"):
            raise ValueError("mode 必须为 segmentation 或 fusion")

        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法解析图片，请上传有效的 JPG/PNG 图片")

        out, detections = self._process_image_bgr(image, mode)

        return self._image_result(
            image,
            out,
            detections,
            mode,
            uuid.uuid4().hex[:12],
        )

    def process_image_file(self, path: str, mode: str):
        """从已落盘的图片文件推理（批量任务用）。"""
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")

        if mode not in ("segmentation", "fusion"):
            raise ValueError("mode 必须为 segmentation 或 fusion")

        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法解析图片文件")

        out, detections = self._process_image_bgr(image, mode)

        return self._image_result(
            image,
            out,
            detections,
            mode,
            uuid.uuid4().hex[:12],
        )

    def start_images_job(self, paths, mode: str) -> str:
        """批量图片任务：逐张推理，每张更新进度，返回 job_id。"""
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")

        job_id = uuid.uuid4().hex[:16]

        with self._job_lock:
            self._jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "result": None,
                "error": None,
                "cancel_requested": False,
            }

        threading.Thread(
            target=self._run_images_job,
            args=(job_id, list(paths), mode),
            daemon=True,
        ).start()

        return job_id

    def _run_images_job(self, job_id: str, paths, mode: str):
        """批量图片任务：逐张推理（大图批处理无收益，逐张更快），逐张更新进度。"""
        results = []
        total = len(paths)

        try:
            for i, path in enumerate(paths):
                if self._jobs[job_id].get("cancel_requested", False):
                    raise JobCancelled()

                result = self.process_image_file(path, mode)
                results.append(result)

                with self._job_lock:
                    job = self._jobs[job_id]
                    job["progress"] = (i + 1) / total
                    job["processed"] = i + 1
                    job["total"] = total

            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "done"
                job["progress"] = 1.0
                job["result"] = {"mode": mode, "results": results}
        except JobCancelled:
            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "cancelled"
                job["error"] = None
        except Exception as exc:  # noqa: BLE001
            with self._job_lock:
                job = self._jobs[job_id]
                job["status"] = "error"
                job["error"] = str(exc)
        finally:
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _video_codec(path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            return ""

        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        codec = "".join(
            chr((fcc >> 8 * i) & 0xFF)
            for i in range(4)
        ).lower()

        return codec

    def _prepare_input(self, raw_path, target_path):
        """把输入视频转成 H.264 MP4（faststart），浏览器必定可播放。

        已是 H.264 或没有 ffmpeg 时直接复用原文件。
        """
        codec = self._video_codec(raw_path)

        ffmpeg = self._find_ffmpeg()

        if codec in ("h264", "avc1") or ffmpeg is None:
            os.replace(raw_path, target_path)
            return

        # 有 NVIDIA GPU 用 NVENC 硬编码转码，否则用 libx264（CPU）
        if self.nvenc_available:
            video_encoder = "h264_nvenc"
            quality_args = ["-preset", "p5", "-cq", "23"]
        else:
            video_encoder = "libx264"
            quality_args = ["-preset", "veryfast", "-crf", "23"]

        transcode_cmds = [
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", raw_path,
                "-c:v", video_encoder,
                *quality_args,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "copy",
                target_path,
            ],
            # 音频流不兼容 mp4 时丢弃音频重试
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", raw_path,
                "-c:v", video_encoder,
                *quality_args,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                target_path,
            ],
        ]

        for cmd in transcode_cmds:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=180,
                )

                if (
                    result.returncode == 0
                    and os.path.exists(target_path)
                    and os.path.getsize(target_path) > 0
                ):
                    os.remove(raw_path)
                    return
            except Exception:  # noqa: BLE001
                continue

            if os.path.exists(target_path):
                os.remove(target_path)

        # 全部失败：直接使用原始文件（管线内仍有处理输出）
        os.replace(raw_path, target_path)

    # ---------------------------------------------------------------
    # 视频编码：优先 NVENC 硬件编码，失败回退 OpenCV（avc1 → mp4v）
    # ---------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg():
        return shutil.which("ffmpeg")

    def _check_nvenc(self):
        ffmpeg = self._find_ffmpeg()

        if not ffmpeg:
            return False

        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=20,
            )

            return "h264_nvenc" in out.stdout
        except Exception:  # noqa: BLE001
            return False

    def _check_nvdec(self):
        ffmpeg = self._find_ffmpeg()

        if not ffmpeg:
            return False

        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=20,
            )

            return "cuda" in out.stdout
        except Exception:  # noqa: BLE001
            return False

    def _open_cv2_writer(self, path, fps, size):
        candidates = ("avc1", "mp4v")

        for codec in candidates:
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*codec),
                fps,
                size,
            )

            if writer.isOpened():
                return writer

        raise RuntimeError("无法创建视频编码器 (avc1/mp4v)")

    def _open_nvenc(self, path, fps, size):
        """通过 ffmpeg h264_nvenc 编码。失败返回 None。"""
        ffmpeg = self._find_ffmpeg()

        width, height = size

        if (
            not ffmpeg
            or width % 2 != 0
            or height % 2 != 0
        ):
            return None

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.2f}",
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-cq", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            path,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:  # noqa: BLE001
            return None

        class _NVENCWriter:
            def __init__(self, process):
                self.proc = process

            def write(self, frame_bgr):
                self.proc.stdin.write(
                    np.ascontiguousarray(frame_bgr).tobytes()
                )

            def release(self):
                try:
                    self.proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass

                self.proc.wait(timeout=120)

                if self.proc.returncode != 0:
                    err = (
                        self.proc.stderr.read()
                        .decode(errors="replace")
                        .strip()
                    )
                    raise RuntimeError(
                        f"NVENC 编码失败: {err or 'unknown error'}"
                    )

        return _NVENCWriter(proc)

    def _open_encoder(self, path, fps, size):
        if self.nvenc_available:
            nv = self._open_nvenc(path, fps, size)

            if nv is not None:
                return nv

        return self._open_cv2_writer(path, fps, size)

    # ---------------------------------------------------------------
    # 视频解码：优先 NVDEC 硬解码（ffmpeg -hwaccel cuda），失败回退 OpenCV
    # ---------------------------------------------------------------

    def _nvd_decoder(self, path, width, height):
        """NVDEC 硬解码：ffmpeg 解码后以 bgr24 原始帧输出到 stdout。"""
        ffmpeg = self._find_ffmpeg()

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-i", path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]

        frame_bytes = width * height * 3

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes * 4,
        )

        try:
            while True:
                buf = b""

                while len(buf) < frame_bytes:
                    chunk = proc.stdout.read(
                        frame_bytes - len(buf)
                    )

                    if not chunk:
                        break

                    buf += chunk

                if len(buf) < frame_bytes:
                    break

                yield np.frombuffer(
                    buf,
                    dtype=np.uint8,
                ).reshape(height, width, 3)
        finally:
            proc.stdout.close()
            proc.wait(timeout=60)

    def _cv2_decoder(self, path):
        cap = cv2.VideoCapture(path)

        def frames():
            try:
                while True:
                    ok, frame = cap.read()

                    if not ok:
                        break

                    yield frame
            finally:
                cap.release()

        return frames()

    def _open_decoder(self, path, width, height):
        """返回 (帧迭代器, 解码器名)。优先 NVDEC，失败自动回退 OpenCV。"""
        if self.nvdec_available:
            gen = self._nvd_decoder(path, width, height)

            try:
                first = next(gen)

                def chain():
                    yield first
                    yield from gen

                return chain(), "nvdec"
            except Exception:  # noqa: BLE001
                pass

        return self._cv2_decoder(path), "opencv"

    @staticmethod
    def _writer_ok(path, expected_frames):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False

        cap = cv2.VideoCapture(path)

        ok = cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return ok and n > 0

    def _run_video_pipeline(
        self,
        input_path,
        output_path,
        mode,
        on_progress=None,
        is_cancelled=None,
    ):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError("无法解析上传的视频，请上传有效的 MP4")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 时长限制已彻底取消（MAX_VIDEO_SECONDS = None），处理完整视频

        if total <= 0:
            cap.release()
            raise ValueError("视频没有可读取的帧")

        cap.release()

        # 主流程：NVDEC 硬解码 + NVENC 硬编码（无 GPU 时自动回退 CPU）
        writer = self._open_encoder(
            output_path,
            fps,
            (width, height),
        )

        frames, decoder_used = self._open_decoder(
            input_path,
            width,
            height,
        )

        encode_ok = True
        cancelled = False
        vehicle_frames = 0
        processed = 0
        start = time.perf_counter()

        batch_size = max(
            1,
            int(getattr(self.cfg, "VIDEO_BATCH_SIZE", 1) or 1),
        )
        batch = []

        try:
            for frame in frames:
                if processed >= total:
                    break

                if is_cancelled is not None and is_cancelled():
                    cancelled = True
                    break

                batch.append(frame)

                if len(batch) < batch_size:
                    continue

                out_frames, vc = self._process_frames(batch, mode)
                vehicle_frames += vc
                batch = []

                for out_frame in out_frames:
                    try:
                        writer.write(out_frame)
                    except (BrokenPipeError, RuntimeError):
                        encode_ok = False
                        break

                if not encode_ok:
                    break

                processed += len(out_frames)

                if on_progress:
                    on_progress(processed, total)

            # 处理剩余不足一批的帧
            if encode_ok and batch:
                out_frames, vc = self._process_frames(batch, mode)
                vehicle_frames += vc

                for out_frame in out_frames:
                    try:
                        writer.write(out_frame)
                    except (BrokenPipeError, RuntimeError):
                        encode_ok = False
                        break

                processed += len(out_frames)

                if on_progress:
                    on_progress(processed, total)
        finally:
            if hasattr(frames, "close"):
                frames.close()

        try:
            writer.release()
        except RuntimeError:
            encode_ok = False

        elapsed = time.perf_counter() - start

        if cancelled:
            raise JobCancelled()

        if (
            not encode_ok
            or processed < total
            or not self._writer_ok(output_path, processed)
        ):
            # 解码/编码失败或帧数不足 → 回退 CPU（OpenCV 解码 + 编码）
            encode_ok, processed, elapsed = self._encode_fallback(
                input_path,
                output_path,
                fps,
                (width, height),
                mode,
                total,
                on_progress,
                is_cancelled,
            )

            decoder_used = "opencv"

        if not encode_ok or processed == 0:
            raise RuntimeError("视频编码失败，请重试")

        avg_ms = (
            elapsed * 1000 / processed
            if processed
            else 0.0
        )

        metrics = {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frame_count": processed,
            "processing_time_s": round(elapsed, 2),
            "avg_inference_time_ms": round(avg_ms, 2),
            "device": self.device,
            "decoder": decoder_used,
            "encoder": "nvenc" if self.nvenc_available else "opencv",
        }

        if mode == "fusion":
            metrics["frames_with_vehicles"] = vehicle_frames

        return {
            "mode": mode,
            "files": {
                "input": os.path.basename(input_path),
                "output": os.path.basename(output_path),
            },
            "metrics": metrics,
            "model_source": self.model_source,
            "yolo": {
                "available": self.yolo is not None,
                "message": "ok" if self.yolo is not None
                else "YOLO model unavailable: 权重缺失",
            },
        }

    def _encode_fallback(
        self,
        input_path,
        output_path,
        fps,
        size,
        mode,
        total,
        on_progress,
        is_cancelled=None,
    ):
        """回退到 OpenCV 编码重新处理。返回 (ok, processed, elapsed)。"""
        writer = self._open_cv2_writer(output_path, fps, size)

        cap = cv2.VideoCapture(input_path)

        processed = 0
        cancelled = False
        start = time.perf_counter()

        batch_size = max(
            1,
            int(getattr(self.cfg, "VIDEO_BATCH_SIZE", 1) or 1),
        )
        batch = []

        try:
            while True:
                ok, frame = cap.read()

                if not ok or processed >= total:
                    break

                if is_cancelled is not None and is_cancelled():
                    cancelled = True
                    break

                batch.append(frame)

                if len(batch) < batch_size:
                    continue

                out_frames, _ = self._process_frames(batch, mode)
                batch = []

                for out_frame in out_frames:
                    writer.write(out_frame)

                processed += len(out_frames)

                if on_progress:
                    on_progress(processed, total)

            # 处理剩余不足一批的帧
            if batch:
                out_frames, _ = self._process_frames(batch, mode)

                for out_frame in out_frames:
                    writer.write(out_frame)

                processed += len(out_frames)

                if on_progress:
                    on_progress(processed, total)
        finally:
            cap.release()

        elapsed = time.perf_counter() - start
        writer.release()

        if cancelled:
            raise JobCancelled()

        ok = self._writer_ok(output_path, processed)

        return ok, processed, elapsed
