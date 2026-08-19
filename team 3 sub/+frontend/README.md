# SegFormer Road Segmentation — Video Web Demo

基于 `team 3 sub/seg分好/` 现有 SegFormer / YOLO Fusion 代码的**视频 Web Demo**：

```text
浏览器上传视频 → FastAPI → Python 模型逐帧推理 → 输出结果视频 → 浏览器播放
```

## 0. 快速启动（一键）

```powershell
# 双击 start.bat，或在项目根目录运行：
.\start.bat
```

脚本会自动完成：创建虚拟环境 → 安装依赖（有 NVIDIA GPU 自动装 CUDA 版 PyTorch）→
下载 YOLO 权重 → 构建前端 → 启动后端，并自动打开浏览器 `http://127.0.0.1:8000`。

**单端口访问**：前端已打包进后端静态服务，无需再单独启动 Vite。
关闭脚本窗口即停止服务。首次启动会自动下载 SegFormer 预训练权重（约 100MB）。

> 前端代码改动后需重新构建：`cd web && npm run build`。
> 开发模式（热更新）仍可：`cd web && npm run dev` 访问 `http://localhost:5173`。

## 目录结构

```text
seg分好/
├─ config/config.py          训练 / 推理路径配置（已修复硬编码绝对路径）
├─ predict.py                SegFormer 道路分割（重构为可复用函数 + CLI）
├─ predict_fusion_yolo.py    SegFormer + YOLO Fusion（同上）
├─ evaluate.py / benchmark.py  已修复导入与路径
├─ server/
│  ├─ main.py                FastAPI 入口（HTTP / 上传 / 静态结果）
│  └─ inference_service.py   模型加载与逐帧推理（启动时加载一次）
├─ web/                      React + TypeScript + Vite + Tailwind 前端
├─ weights/yolo/best.pt      YOLO 权重（从仓库下载，见下文）
├─ data/test/                示例图片与测试视频
└─ requirements.txt          后端依赖
```

## 1. 后端启动（FastAPI，端口 8000）

```powershell
# ① 创建虚拟环境
python -m venv .venv

# ② 安装依赖（有 NVIDIA GPU 时先装 CUDA 版 torch，推理默认走 GPU）
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt

# ③ 启动服务（模型在启动时加载）
.\.venv\Scripts\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

无 GPU 时把第 ② 步的 torch 安装改为 CPU 版即可：

```powershell
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`config/config.py` 的 `DEVICE` 会自动选择：`"cuda" if torch.cuda.is_available() else "cpu"`，
无需手动修改；`/api/health` 返回的 `device` 字段可确认当前实际使用的设备。

国内网络无法直连 HuggingFace 时，先设置镜像再启动：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

首次启动会自动下载 SegFormer-B0 预训练权重（约 100MB，缓存在用户目录），之后直接加载。

## 2. YOLO 权重

本仓库 `seg分好/` 目录本身没有权重；YOLO 权重在仓库
`team submissions second week/YOLOv11s-P2/weights/best.pt`（64MB，YOLOv11s-P2，自定义 15 类）。

```powershell
New-Item -ItemType Directory -Force weights/yolo | Out-Null
curl.exe -L -o weights/yolo/best.pt "https://raw.githubusercontent.com/yyh061129vic-debug/driver-vision-risk-warning-system/main/team%20submissions%20second%20week/YOLOv11s-P2/weights/best.pt"
```

权重缺失时：SegFormer 模式仍正常；Fusion 模式返回 `YOLO model unavailable` 并退回 SegFormer 结果。
（SegFormer 同理：若放入训练好的 `weights/best_segformer_road.pth` + `models/segformer-b0/` 则优先加载，否则使用 HF 预训练兜底，均为真实推理。）

## 3. 前端启动（Vite，端口 5173）

```powershell
cd web
npm install
npm run dev
```

打开浏览器访问 `http://localhost:5173/`（开发服务器已将 `/api` 代理到后端 8000 端口）。

## 4. 使用流程

1. 选择 **视频 / 图片** 类型，上传 MP4 视频或 JPG/PNG 图片（视频**大小、时长、分辨率均无限制**）
2. 选择模式：
   - **SegFormer**：道路分割
   - **SegFormer + YOLO**：YOLO 检测车辆，从 road mask 中剔除（Fusion），并绘制车辆框
3. 点击 **Start Analysis**
4. 查看结果：视频为 Original / Processed 并排播放；图片为原图 / 结果图对比（Fusion 含红色车辆框）
5. 指标：Resolution、FPS、Frame Count、Processing Time、Avg Inference、Device（Decoder/Encoder、Frames w/ Vehicles）
6. 右上角切换 Light / Dark 模式

## 5. API

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 服务与模型状态（含 YOLO 是否可用） |
| POST | `/api/process-video` | multipart 上传 `file` + 表单字段 `mode`，立即返回 `job_id`，后台线程处理 |
| GET | `/api/process-status/{job_id}` | 轮询任务进度（`progress` 0-1，`status`：processing/done/error/cancelled，完成后含 `result`） |
| POST | `/api/process-cancel/{job_id}` | 终止正在运行的任务 |
| POST | `/api/process-image` | 上传单张图片（`file` + `mode`），同步返回结果图路径与车辆数 |
| POST | `/api/process-images` | 批量上传多张图片（多个 `files` + `mode`），返回 `job_id`，轮询进度按已处理张数推进 |
| GET | `/api/results/{filename}` | 结果文件（原图/视频 / 结果图/视频） |

POST 返回：

```json
{ "job_id": "abc123", "status": "processing" }
```

`/api/process-status/{job_id}` 返回（处理中）：

```json
{ "status": "processing", "progress": 0.45, "processed": 36, "total": 80 }
```

完成后返回：

```json
{
  "status": "done",
  "progress": 1.0,
  "result": {
    "mode": "fusion",
    "files": { "input": "input_xxx.mp4", "output": "output_xxx.mp4" },
    "metrics": {
      "resolution": "640x360",
      "fps": 20.0,
      "frame_count": 80,
      "processing_time_s": 3.3,
      "avg_inference_time_ms": 41.66,
      "device": "cuda"
    },
    "model_source": "pretrained-fallback",
    "yolo": { "available": true, "message": "ok" }
  }
}
```

## 6. 说明

- **逐帧推理**：服务启动时加载 SegFormer（与 YOLO）一次，视频循环逐帧调用推理函数，不重复加载模型。推理全部在 GPU（`/api/health` 返回 `device: cuda`）。
- **批处理加速**：视频按 `VIDEO_BATCH_SIZE`（默认 8）攒帧批量推理，SegFormer 吞吐提升约 27%（640x360 实测 12.4→9.1 ms/帧）。YOLO 为小模型，批处理无额外收益，逐帧推理。
- **时长不限**：`MAX_VIDEO_SECONDS = None`，处理完整视频不截断。单帧总耗时 ≈ SegFormer + YOLO + 编码（640x360 实测约 40-45 ms/帧）。
- **上传无限制**：视频流式落盘（分块写入，不占内存），大小无上限；分辨率兼容至 4K 级别（超过时逐帧插值）。
- **GPU 全硬件管线（自动检测）**：服务启动时检测是否有 NVIDIA GPU（`torch.cuda.is_available()`）——有则使用 **NVDEC 硬解码 + NVENC 硬编码 + GPU 推理**（视频解析与编码都不走 CPU）；无 GPU 则自动回退 OpenCV 解码 + CPU 编码。指标中 `decoder`（`nvdec` / `opencv`）与 `encoder`（`nvenc` / `opencv`）字段可在前端直接查看。
- **输入视频转码**：上传的视频若不是 H.264，后端会自动用 NVENC 转码为 H.264（faststart），保证浏览器一定能播放原视频与结果视频。
- **任务式处理**：`POST /api/process-video` 立即返回 `job_id`，前端每 0.8s 轮询进度并显示"x / y 帧 · 百分比"进度条。
- **前端直连后端**：媒体与 API 默认直连 `http://127.0.0.1:8000`（CORS 已放开），避免开发代理对流媒体的干扰。如需修改，在 `web/.env` 中设置 `VITE_API_BASE`。
- **车辆类别**：仓库 YOLO 权重为自定义 15 类，车辆类别 `(0, 5, 6, 7, 8, 14)`（bus/truck/motor/car/train/vehicle）已配置在 `config/config.py` 的 `YOLO_VEHICLE_CLASSES`。
- **CLI**：`python predict.py` / `python predict_fusion_yolo.py`（读取 `data/test/`，输出到 `results/masks/`）。
- 指标全部为真实计算，无伪造（不包含 Accuracy / mIoU / Risk 等）。

## 7. 已修复的原有问题

- `config/config.py`：移除 `/home/epfl/...` 硬编码路径，改为项目相对路径
- `predict.py` / `predict_fusion_yolo.py` / `evaluate.py` / `benchmark.py`：修正 `configs` → `config`、`datasets` → `dataset` 导入错误
- `benchmark.py`：修正 `Config()` 实例化、硬编码模型路径
- `detect_vehicle`：YOLO 类别过滤从硬编码 COCO `[2,3,5,7]` 改为配置项 `YOLO_VEHICLE_CLASSES`（适配仓库自定义 15 类权重）
