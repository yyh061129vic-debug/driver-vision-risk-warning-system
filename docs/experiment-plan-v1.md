# 第一版实验方案冻结说明

## 1. 冻结结论

第一版可行驶区域实验方案已于 2026-07-16 冻结，版本为 `phase1_drivable_area_v1 / 1.0.0`。本版本是推理评测基线，不在本地训练、微调或拟合阈值。冻结范围包括数据划分、模型 revision、权重哈希、模型输入尺寸、推理决策规则、指标定义、随机种子和性能测试口径。

冻结后的关键因素不得在原文件中静默修改。若更换数据、模型、输入尺寸、标签映射或指标定义，必须创建新的实验版本，并保留本版本用于对照。

## 2. 数据划分

| 分区 | 数据 | 数量 | 用途 |
| --- | --- | ---: | --- |
| Train | 无本地样例 | 0 | 使用固定预训练权重，不进行本地训练 |
| Development smoke | Lost and Found train，3 个序列 | 6 | 仅检查读取、推理和可视化流程，不选指标或阈值 |
| Holdout evaluation | Lost and Found test，3 个序列 | 4 | 单独报告非商业小障碍物场景指标 |
| Holdout evaluation | RoadObstacle21 public validation | 30 | 单独报告道路障碍物公开验证指标 |

总计 40 张本地样例，其中 6 张只做开发冒烟检查，34 张进入冻结评测。Lost and Found 开发序列与测试序列完全分离；RoadObstacle21 的 30 张公开验证图全部进入留出评测，不用于阈值调整。划分清单见 `data/indexes/drivable_area_v1_split.yaml`。

由于当前 SegFormer 权重已经由上游在 Cityscapes 上训练，本地 `train` 分区明确为零。这样可以避免把 Lost and Found 或 RoadObstacle21 的评测数据误当训练数据，也不会把任务四人为挑选的可视化样例包装成完整训练集。

## 3. 冻结模型

| 字段 | 固定值 |
| --- | --- |
| 模型 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` |
| 架构 | `SegformerForSemanticSegmentation` |
| Revision | `21b3847fae21ddee674abd31129307b6a1235bd9` |
| 权重 SHA-256 | `027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c` |
| 道路类别 | Cityscapes class 0：`road` |
| 参数状态 | 全部冻结，不训练、不微调 |
| 决策规则 | 19 类 logits 逐像素 argmax，无额外置信度阈值 |

模型及许可登记继续复用 `configs/models/segformer_cityscapes_cpu.yaml`、`checkpoints/index.yaml` 和对应许可快照。该文件保留任务六冻结时的 CPU 对照；交互式 Demo 的默认 `configs/models/segformer_cityscapes.yaml` 后续可使用 GPU，但不得覆盖 V1 CPU 基线记录。

## 4. 冻结输入与输出

模型实际图像处理器已在本地验证为 `RGB / uint8 / batch=1 / 512×512`。预处理强制缩放到 512×512，不保持宽高比，随后按 ImageNet mean/std 归一化。模型 logits 使用双线性插值恢复到原始输入分辨率，再计算类别、掩码、边界和指标。

模型仓库名称包含 `1024-1024`，但当前固定 revision 的 `preprocessor_config.json` 明确记录 `size: 512`，实际处理器张量形状也验证为 `1×3×512×512`。因此 V1 冻结的是实际运行尺寸 512×512，不以仓库名称推断输入尺寸。

## 5. 冻结指标

主要质量指标按数据集分别累计混淆矩阵并报告，不把两个不同数据域混合成唯一“最好成绩”。所有指标忽略 void 像素。

1. `Binary mIoU`：道路 IoU 与非道路 IoU 的算术平均。
2. `Road IoU`：道路预测与道路真值的交集除以并集。
3. `Boundary F-score`：在原图分辨率上计算道路边界的对称 F1，四邻域边界，匹配容差固定为 3 像素。

诊断指标包括 Road Precision、Road Recall、Road Dice、道路覆盖率和道路区域平均置信度。性能指标包括平均延迟、P95 延迟和吞吐 FPS；以 CPU、float32、batch=1 为当前本地基线，预热 5 帧，对 34 张留出图像执行一次计时。计时从图像处理器开始到类别图产生，排除模型加载、文件读取、可视化和文件写入。

需求文档没有给出可行驶区域数值合格线，目标 GPU 和延迟预算也仍是待确认项。因此本版本冻结“评什么、如何计算”，不虚构 IoU、BF1 或实时性门槛。数值验收线必须由产品、安全和目标硬件负责人确认后进入新版本。

## 6. 报告与失败案例

正式执行 V1 时必须分别报告 Lost and Found 与 RoadObstacle21 结果，并至少整理以下失败类型：障碍物被预测为道路、道路被预测为非道路、边界偏移、数据域或图像质量变化。当前样例不具备可靠的距离、天气和时段结构化字段，相关分层一律记录为 `unknown`，禁止仅凭画面主观猜测。

本方案只证明第一版分割基线可复现，不代表未知异常检测、距离估计、风险状态机、目标 GPU 实时性或车规安全已经完成。

## 7. 校验命令

```powershell
python scripts/validate_experiment_plan.py
```

校验器会检查冻结状态、版本引用、模型 revision 和哈希、512×512 预处理、三项主指标、数据数量、样例/序列泄漏、原图与标注完整性，以及未确认门槛是否保持待定。
