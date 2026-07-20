# 任务 4：道路与异常样例可视化

## 1. 验收结论

启动清单任务 4 要求输出不少于 20 张样例可视化，并覆盖道路与异常场景。本次在本地生成 20 张标注叠加图：Lost and Found 10 张、RoadObstacle21 10 张；每张图均同时包含道路区域和异常/障碍物区域，另生成 1 张总览图。

| 数据集 | 样例数 | 覆盖范围 | 用途限制 |
| --- | ---: | --- | --- |
| Lost and Found | 10 | 6 个序列，train/test 均覆盖 | 仅本地非商业验证，不得再分发原图或修改图 |
| RoadObstacle21 | 10 | 10 个 validation 场景，含极小与较大障碍 | 可视化与评测，不作为训练语料 |
| 合计 | 20 | 道路、近远障碍、不同障碍尺寸 | 所有图像产物保留在 Git 忽略目录 |

程序校验确认 20 张图像均可读取，且每张图的道路像素数和障碍像素数均大于 0。样例清单和输入文件 SHA256 由 `outputs/task4_sample_visualizations/summary.json` 记录。

## 2. 可视化约定

- 绿色半透明区域：道路或可行驶区域。
- 红色高亮区域：道路障碍物或未知异常区域。
- 白色轮廓：障碍物边界，用于增强小目标可见性。
- 未标注或 ROI 外区域：保持原图，不参与颜色叠加。

Lost and Found 使用 `labelIds == 1` 表示道路，`2..254` 表示一个或多个障碍实例；RoadObstacle21 使用 `0` 表示道路、`1` 表示障碍、`255` 表示 ROI 外区域。映射规则集中在 `configs/data/task4_samples.yaml`，未硬编码到核心业务逻辑。

## 3. 数据与产物位置

- 样例配置：`configs/data/task4_samples.yaml`
- 可版本化索引：`data/indexes/task4_samples.jsonl`
- 原始数据：`data_raw/lost-and-found/` 与 `data_raw/segment-me-if-you-can/`
- 单张叠加图：`outputs/task4_sample_visualizations/samples/`
- 总览图：`outputs/task4_sample_visualizations/contact-sheet.png`
- 结果摘要：`outputs/task4_sample_visualizations/summary.json`

原始数据和可视化结果均由目录忽略规则排除，不进入 Git。Lost and Found 没有下载完整的 5.8 GB 左图压缩包，而是通过官方压缩包 HTTP Range 只提取清单中的 10 张图；其完整标注包和 RoadObstacle21 官方包均已完成哈希校验。

## 4. 本地复现

安装依赖：

```bash
python -m pip install -e ".[dev]"
```

原始包就绪后，按清单提取 Lost and Found 图像、生成叠加图并验收：

```bash
python scripts/download_task4_samples.py
python scripts/visualize_dataset_samples.py
python scripts/validate_task4_visualizations.py
```

若官方 Hugging Face 入口在当前网络下不能稳定跟随重定向，可从该官方入口取得临时签名地址并通过 `--lost-and-found-image-archive-url` 传入；签名地址不得写入仓库。

## 5. 许可边界与局限

- Lost and Found 的原图、修改图和可视化不得对外再分发，本任务仅在本地工作区保存。
- RoadObstacle21 样例来自 validation 集，仅用于可视化与后续评测，不参与训练或阈值拟合。
- 当前 20 张样例用于工程链路和标注质量检查，不能代表完整数据分布，也不能作为模型性能结论。
- 连续帧样例仍属于同一序列；后续实验划分必须以序列或场景为单位，禁止相邻帧跨集合泄漏。
