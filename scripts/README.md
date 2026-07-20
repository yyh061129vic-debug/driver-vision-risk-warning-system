# 脚本目录

后续下载、转换、校验、可视化、训练、评估和 CARLA 场景脚本放在此目录。脚本必须从配置读取路径与参数，并对错误返回明确的非零退出码。

`verify_layout.py` 是仓库结构验收脚本；`validate_dataset_registry.py` 检查 7 个候选数据集及本地启用状态；`validate_environment_baseline.py` 检查环境快照；`download_task4_samples.py` 最小化提取任务 4 输入；`visualize_dataset_samples.py` 生成道路/障碍叠加图和索引；`validate_task4_visualizations.py` 检查 20 张图、总览图和像素覆盖。

任务 5 使用 `download_segmentation_model.py` 下载并校验固定版本 SegFormer 权重，`run_segmentation_demo.py` 接收单张图像或视频并输出可行驶区域叠加，`validate_segmentation_demo.py` 检查模型登记、权重哈希、掩码、边界、置信度图、叠加图和运行记录。

任务 6 使用 `validate_experiment_plan.py` 检查 V1 冻结状态、模型 revision、512×512 输入、主要指标、样例数量、序列泄漏以及本地 40 张样例与模型处理器完整性。`--metadata-only` 模式不要求保留 Git 忽略的原始数据和权重。
