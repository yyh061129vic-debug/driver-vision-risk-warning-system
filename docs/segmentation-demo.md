# 任务 5：可行驶区域分割 Demo

## 1. 验收结论

任务 5 已在本地跑通。程序可接收图像或视频，输出可行驶区域叠加；单图模式同时保存像素掩码、边界、道路置信度图和可机器校验的运行记录。

本次真实验收输入为 RoadObstacle21 的 `validation_34.webp`（1920×1080）。叠加结果中的道路为绿色、边界为白色，前景犬只未被划入可行驶区域。自动校验确认所有输出尺寸与输入一致、掩码非空且非全图、叠加像素发生有效变化，文件哈希与 `result.json` 一致。

## 2. 固定模型

| 项目 | 固定值 |
| --- | --- |
| 模型 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` |
| 架构 | `SegformerForSemanticSegmentation` |
| Revision | `21b3847fae21ddee674abd31129307b6a1235bd9` |
| 道路类别 | Cityscapes 类别 0：`road` |
| 权重文件 | `pytorch_model.bin`，14,957,601 bytes |
| 权重 SHA-256 | `027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c` |
| 后处理 | 对 19 类 logits 双线性上采样后取 argmax；未人为引入置信度阈值 |
| 默认设备 | CUDA / RTX 4070 Laptop GPU / float32 |
| CPU 对照 | `configs/models/segformer_cityscapes_cpu.yaml`，12 个 PyTorch 线程 |

模型配置见 `configs/models/segformer_cityscapes.yaml`，权重登记见 `checkpoints/index.yaml`。下载脚本只访问固定 revision，并在可用前验证模型结构、道路标签、文件大小和 SHA-256。

模型许可证为 NVIDIA Source Code License for SegFormer，只允许非商业研究和评估。许可快照及用途边界见 `metadata/licenses/segformer-b0-cityscapes-2026-07-16.md`；本 Demo 不构成商业使用或车规安全许可。

## 3. 本机实测结果

| 指标 | 结果 |
| --- | ---: |
| 输入分辨率 | 1920×1080 |
| GPU 模型前向与张量后处理延迟 | 418.599 ms |
| CPU 对照延迟 | 703.039 ms |
| 道路像素 | 537,623 |
| 道路覆盖率 | 25.927% |
| 道路区域平均置信度 | 97.643% |
| PyTorch | 2.13.0+cu130 |
| TorchVision | 0.28.0+cu130 |
| Transformers | 5.14.0 |

这里的 GPU 延迟经过 CUDA 同步后统计，覆盖图像处理器、模型前向、logits 上采样、softmax 和类别图计算，不包括模型初始化、文件读取、可视化与磁盘写入。CPU 数字保留为任务 6 冻结时的对照；两者都不代表尚待确认的最终目标硬件验收结果。

默认 GPU 单图输出位于 `outputs/task5_segmentation_demo/gpu/`：

- `overlay.png`：可行驶区域与边界叠加；
- `drivable-mask.png`：二值道路掩码；
- `drivable-boundary.png`：掩码内边界；
- `road-confidence.png`：道路类别概率的 8 位灰度图；
- `result.json`：输入、模型 revision、权重哈希、运行版本、指标和输出哈希。

## 4. 复现命令

在项目根目录执行：

```powershell
python -m pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,inference]"
python scripts/download_segmentation_model.py
python scripts/run_segmentation_demo.py `
  --input "data_raw/segment-me-if-you-can/road-obstacle-21/extracted/dataset_ObstacleTrack/images/validation_34.webp" `
  --output "outputs/task5_segmentation_demo/gpu"
python scripts/validate_segmentation_demo.py
```

也可通过安装后的命令运行：

```powershell
driver-vision-risk segment --input <图像或视频路径> --output <输出目录>
```

## 1024 输入与未知异常候选

`configs/models/segformer_cityscapes_anomaly.yaml` 是独立的异常检测实验配置，不修改冻结的
`segformer_cityscapes_cpu.yaml`。它把模型预处理输入提升到 1024×1024，并默认使用能量分数；
把 `anomaly.score_method` 改为 `msp` 即可切换到 `1 - MSP`。

```powershell
python scripts/run_segmentation_demo.py `
  --input "data_raw/segment-me-if-you-can/road-obstacle-21/extracted/dataset_ObstacleTrack/images/validation_34.webp" `
  --output "outputs/task5_segmentation_anomaly/validation_34" `
  --config "configs/models/segformer_cityscapes_anomaly.yaml"
```

单图输出新增 `anomaly-heatmap.png`。`result.json` 的
`anomaly_detection.regions` 是风险状态机接口：每个元素包含右下角开区间的
`bbox_xyxy`、`area_pixels` 和 `mean_anomaly_score`。候选像素必须同时满足“模型预测为道路”
和“异常分数高于阈值”，随后经过连通域和最小面积过滤。

候选提取前还会按 `anomaly.road_mask_erosion_pixels` 腐蚀预测道路掩码。当前实验值为
5 个原图像素，用于排除紧贴道路边界的路沿响应；路外植被、围栏和天空即使热力图为红色，
也不会进入 `anomaly_detection.regions`。全图热力图仅供分析能量分布，不代表全图都参与风险判定。

模型在 argmax 类别交界处天然犹豫，因此程序还会检测完整类别图的交界线，按
`anomaly.class_boundary_suppression_pixels` 膨胀后从候选掩码排除。当前实验半径为 8 个
原图像素。实现采用布尔掩码排除，而不是把 Energy 数值写成零；因为当前能量阈值为负数，
数值零反而会被解释为更可疑。

输出同时保留 `anomaly-raw-heatmap.png` 和 `anomaly-heatmap.png`：前者展示未经抑制的
全图能量，供模型诊断；后者把路外区域及膨胀后的类别边界带显示为黑色，只展示真正参与
风险候选提取的分数。风险状态机只读取后者对应的规则和 `regions`。

当前能量阈值 `-1.0` 与最小面积 `256` 仅为本地样例首轮工程值，状态为
`experimental_uncalibrated`，不能当作产品安全阈值或验收门槛。

## 5. 已知边界

- 模型只在 Cityscapes 上做过微调，RoadObstacle21、乡村道路、极端天气和夜间图像存在域偏移。
- 语义道路分割不是障碍物检测；本例中犬只被排除不等于所有未知障碍物都会被排除。
- 当前采用 argmax 语义标签，没有凭空设定安全阈值；风险阈值必须在后续实验方案和目标场景中校准。
- 视频入口已实现逐帧叠加和 MP4 输出，本次任务验收实际运行的是单张图像。
- 当前已验证 RTX 4070 GPU 推理，但不能据此宣称满足最终目标硬件实时性、功能安全或车规部署要求。
