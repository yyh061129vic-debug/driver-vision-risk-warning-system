# PIDNet 模型提交说明

这个目录只提交 `PIDNet-S` 道路分割模型相关源码，目标是做到：

1. 不依赖仓库主流程其他模块也能单独运行；
2. 可以直接下载固定版本模型资产；
3. 可以对单张图片或单个视频完成道路分割推理并输出叠加结果。

## 目录结构

```text
team 3 sub/pidnet/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ pidnet_s_cityscapes.yaml
├─ checkpoints/
│  └─ pidnet-s-cityscapes-onnx-float/   # 运行下载脚本后生成
├─ download_pidnet_model.py
└─ run_pidnet_demo.py
```

## 作用说明

`PIDNet-S` 是一个偏实时的语义分割模型。这里提交的是它在本项目里的道路分割版本，使用固定的 ONNX 资产，只提取 `road` 类别作为可行驶区域。

相比项目原有的 `SegFormer` 主线，这个版本更适合作为实时视频道路分割分支，后续可以继续和障碍物检测、未知风险模块做并行融合。

## 运行环境

- Python 3.11
- Windows / Linux 均可
- 推荐 GPU：安装 `onnxruntime-gpu`
- 仅 CPU 也可以运行，但速度会更慢

## 安装依赖

```powershell
python -m pip install -r "team 3 sub/pidnet/requirements.txt"
```

## 下载模型

```powershell
python "team 3 sub/pidnet/download_pidnet_model.py"
```

下载完成后，模型文件会放到：

```text
team 3 sub/pidnet/checkpoints/pidnet-s-cityscapes-onnx-float/
```

## 运行图片 demo

```powershell
python "team 3 sub/pidnet/run_pidnet_demo.py" --input "D:\bdd100k\100k\val\b1c66a42-6f7d68ca.jpg" --output "team 3 sub/pidnet/outputs/image_demo"
```

## 运行视频 demo

```powershell
python "team 3 sub/pidnet/run_pidnet_demo.py" --input "D:\bdd100k\videos\val\local_demo_road_video.mp4" --output "team 3 sub/pidnet/outputs/video_demo"
```

## 输出内容

图片输入会输出：

- `drivable-mask.png`
- `drivable-boundary.png`
- `road-confidence.png`
- `overlay.png`
- `result.json`

视频输入会输出：

- `overlay.mp4`
- `result.json`

## 已知边界

- 当前版本使用 Cityscapes 导出的固定 ONNX 资产，不是这次训练得到的权重导出版；
- 当前只提取 `road` 类别，不包含车道线、障碍物或风险判断逻辑；
- 如果机器上同时安装了 `torch`，Windows 下会自动预加载 CUDA DLL，供 `onnxruntime-gpu` 复用。
