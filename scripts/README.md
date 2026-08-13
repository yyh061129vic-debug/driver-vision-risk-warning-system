# 脚本目录

后续下载、转换、校验、可视化、训练、评估和 CARLA 场景脚本放在此目录。脚本必须从配置读取路径与参数，并对错误返回明确的非零退出码。

`verify_layout.py` 是仓库结构验收脚本；`validate_dataset_registry.py` 检查 7 个候选数据集及本地启用状态；`validate_environment_baseline.py` 检查环境快照；`manage_datasets.py` 统一处理公开数据下载与本地数据连接；`download_task4_samples.py` 最小化提取任务 4 输入；`visualize_dataset_samples.py` 生成道路/障碍叠加图和索引；`validate_task4_visualizations.py` 检查 20 张图、总览图和像素覆盖。

任务 5 使用 `download_segmentation_model.py` 下载并校验固定版本 SegFormer 权重，`run_segmentation_demo.py` 接收单张图像或视频并输出可行驶区域叠加，`validate_segmentation_demo.py` 检查模型登记、权重哈希、掩码、边界、置信度图、叠加图和运行记录。

`download_pidnet_model.py` 下载并校验固定版本 `PIDNet-S` ONNX 资产，`run_pidnet_demo.py` 复用统一 CLI 但固定 `PIDNet-S` 配置，`validate_pidnet_demo.py` 检查 `PIDNet-S` 的模型登记、资产哈希、掩码、边界、置信度图、叠加图和运行记录。若本机只安装了 CPU 版 `onnxruntime`，`PIDNet-S` 会自动退回 `CPUExecutionProvider`；若额外安装支持 CUDA 的 `onnxruntime` 发行版，则会按配置优先选择 `CUDAExecutionProvider`。现在也可以直接通过主流程命令切换模型：

```powershell
driver-vision-risk segment --model pidnet --input <图像或视频路径> --output outputs/task5_pidnet_demo/mainflow
```

任务 6 使用 `validate_experiment_plan.py` 检查 V1 冻结状态、模型 revision、512×512 输入、主要指标、样例数量、序列泄漏以及本地 40 张样例与模型处理器完整性。`--metadata-only` 模式不要求保留 Git 忽略的原始数据和权重。

`manage_datasets.py` 的典型用法：

```powershell
python scripts/manage_datasets.py status
python scripts/manage_datasets.py init-connections
python scripts/manage_datasets.py download-public --dataset-id segment-me-if-you-can
python scripts/manage_datasets.py connect --dataset-id bdd100k --image-dir "D:\datasets\bdd100k\100k\val" --label-dir "D:\datasets\bdd100k\labels\drivable\masks\val"
```
