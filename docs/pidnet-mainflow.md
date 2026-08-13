# PIDNet 主流程接入说明

## 为什么要加 PIDNet

当前项目的道路分割主线原本以 `SegFormer` 为主。`SegFormer` 的精度基础比较稳，但在车载视频场景里，实时性同样重要，尤其是后续还要继续叠加障碍物检测、未知风险分支和预警逻辑。

这次把 `PIDNet-S` 接进主流程，核心意义有三点：

1. 给主流程补一条更偏实时部署的道路分割分支。
2. 为后续 `SegFormer + PIDNet` 的并行或决策级融合保留统一入口。
3. 让项目里“研究训练结果”和“工程推理主流程”不再割裂，后续视频 demo、联调和 benchmark 都能直接复用。

## 它是什么

`PIDNet` 是面向实时语义分割设计的一类网络。当前仓库接入的是 `PIDNet-S` 的 ONNX 推理版本，用 Cityscapes 导出的 19 类 logits，在本项目里先只提取 `road` 类别，作为可行驶区域分割结果。

当前实现位置：

- 主流程 CLI：`src/driver_vision_risk/cli.py`
- 推理入口：`src/driver_vision_risk/inference/drivable_area.py`
- 模型封装：`src/driver_vision_risk/models/pidnet.py`
- 固定配置：`configs/models/pidnet_s_cityscapes.yaml`
- 资产下载：`scripts/download_pidnet_model.py`

## 怎么接到主流程

现在主流程命令 `driver-vision-risk segment` 已支持直接按模型名切换，不需要再手写配置路径。

默认仍然是 `SegFormer`：

```powershell
driver-vision-risk segment --input <图像或视频> --output outputs/segformer_run
```

切到 `PIDNet`：

```powershell
driver-vision-risk segment --model pidnet --input <图像或视频> --output outputs/pidnet_run
```

如果需要自定义配置，仍然可以显式传 `--config`，此时会覆盖 `--model` 的默认映射：

```powershell
driver-vision-risk segment --model pidnet --config configs/models/pidnet_s_cityscapes.yaml --input <图像或视频> --output outputs/custom_pidnet_run
```

## 使用前准备

1. 安装推理依赖：

```powershell
python -m pip install -e ".[dev,inference]"
```

2. 下载 PIDNet 资产：

```powershell
python scripts/download_pidnet_model.py
```

3. 运行主流程：

```powershell
driver-vision-risk segment --model pidnet --input "D:\bdd100k\videos\val\local_demo_road_video.mp4" --output outputs/pidnet_mainflow_demo
```

## 当前适用边界

- `SegFormer` 仍然保留为默认主线，不被替换。
- `PIDNet-S` 当前更适合作为实时道路分割分支和视频演示分支。
- 当前工程内 `PIDNet` 主流程接入使用的是 ONNX 推理资产；训练闭环和评测闭环已经另外打通，可继续迭代更优权重。

## 为什么这次接入是值得的

这次训练和 benchmark 的结果已经证明，`PIDNet` 相比原项目 `SegFormer` 至少在实时性上有明确提升：统一对比集上平均延迟下降约 `21%`，FPS 提升约 `1.27x`。这使它很适合承担后续视频流场景里的快速道路分割角色，而 `SegFormer` 可以继续作为更稳的精度基线。
