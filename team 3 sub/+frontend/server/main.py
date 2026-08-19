import os
import shutil
import sys
import uuid
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 保证 server 目录之外的项目模块可被导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import Config  # noqa: E402
from server.inference_service import InferenceService  # noqa: E402

app = FastAPI(title="SegFormer Video Segmentation Web Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型在服务启动时加载（首次启动会从 HuggingFace 下载预训练权重）
service = InferenceService()

ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 上传大小无限制：视频流式落盘，不占用内存。


@app.get("/api/health")
def health():
    return {
        "status": "ok" if service.ready else "error",
        "model_source": service.model_source,
        "device": service.device,
        "yolo_available": service.yolo is not None,
        "message": service.error,
    }


@app.post("/api/process-video")
def process_video(
    file: UploadFile = File(...),
    mode: str = Form("segmentation"),
):
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=service.error or "模型未就绪",
        )

    if mode not in ("segmentation", "fusion"):
        raise HTTPException(
            status_code=400,
            detail="mode 必须为 segmentation 或 fusion",
        )

    if not file.filename or not file.filename.lower().endswith(
        ALLOWED_VIDEO_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="仅支持 MP4 / MOV / AVI / MKV 视频",
        )

    # 视频流式落盘（分块写入，不占内存），大小无限制
    token = uuid.uuid4().hex[:12]
    raw_path = os.path.join(Config.RESULT_DIR, f"upload_{token}.mp4")

    try:
        with open(raw_path, "wb") as out:
            shutil.copyfileobj(
                file.file,
                out,
                length=1024 * 1024,
            )
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise HTTPException(
            status_code=400,
            detail=f"保存上传文件失败: {exc}",
        ) from exc
    finally:
        file.file.close()

    try:
        job_id = service.start_video_job(raw_path, mode)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise HTTPException(
            status_code=500,
            detail=f"视频处理失败: {exc}",
        ) from exc

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/process-status/{job_id}")
def process_status(job_id: str):
    job = service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在或已过期",
        )

    return job


@app.post("/api/process-cancel/{job_id}")
def process_cancel(job_id: str):
    if not service.cancel_job(job_id):
        raise HTTPException(
            status_code=404,
            detail="任务不存在",
        )

    return {"status": "cancelling"}


@app.post("/api/process-image")
def process_image(
    file: UploadFile = File(...),
    mode: str = Form("segmentation"),
):
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=service.error or "模型未就绪",
        )

    if mode not in ("segmentation", "fusion"):
        raise HTTPException(
            status_code=400,
            detail="mode 必须为 segmentation 或 fusion",
        )

    if not file.filename or not file.filename.lower().endswith(
        ALLOWED_IMAGE_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="仅支持 JPG / PNG / WEBP / BMP 图片",
        )

    try:
        data = file.file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"读取文件失败: {exc}",
        ) from exc
    finally:
        file.file.close()

    try:
        return service.process_image(data, mode)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"图片处理失败: {exc}",
        ) from exc


@app.post("/api/process-images")
def process_images(
    files: List[UploadFile] = File(...),
    mode: str = Form("segmentation"),
):
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=service.error or "模型未就绪",
        )

    if mode not in ("segmentation", "fusion"):
        raise HTTPException(
            status_code=400,
            detail="mode 必须为 segmentation 或 fusion",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="请至少上传一张图片",
        )

    # 批量图片流式落盘，逐张处理并在任务中推进度
    paths = []

    try:
        for f in files:
            if not f.filename or not f.filename.lower().endswith(
                ALLOWED_IMAGE_EXTENSIONS
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的图片格式: {f.filename}",
                )

            token = uuid.uuid4().hex[:12]
            path = os.path.join(
                Config.RESULT_DIR,
                f"upload_{token}.img",
            )

            with open(path, "wb") as out:
                shutil.copyfileobj(
                    f.file,
                    out,
                    length=1024 * 1024,
                )

            f.file.close()
            paths.append(path)

        job_id = service.start_images_job(paths, mode)
    except Exception as exc:  # noqa: BLE001
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"批量图片处理失败: {exc}",
        ) from exc

    return {"job_id": job_id, "status": "processing"}


# 结果文件静态服务（原视频 / 结果视频）: GET /api/results/{filename}
os.makedirs(Config.RESULT_DIR, exist_ok=True)

app.mount(
    "/api/results",
    StaticFiles(directory=Config.RESULT_DIR),
    name="results",
)

# 生产模式：若前端已构建（web/dist），由后端直接托管，单端口即可访问整个应用。
# 放在最后挂载，保证 /api/* 路由优先匹配。
WEB_DIST = os.path.join(ROOT, "web", "dist")

if os.path.isdir(WEB_DIST):
    app.mount(
        "/",
        StaticFiles(directory=WEB_DIST, html=True),
        name="web",
    )
