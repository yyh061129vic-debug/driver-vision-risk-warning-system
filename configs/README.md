# 配置目录

配置文件必须可版本化、可审查，并避免出现个人机器的绝对路径或密钥。

- `paths.yaml`：仓库内标准路径。
- `system.yaml`：状态枚举、模块开关和待确认参数。
- `data/datasets.yaml`：数据候选、登记文件和启用列表。
- `models/segformer_cityscapes.yaml`：任务 5 默认 CUDA Demo 的模型版本、道路类别、运行设备和可视化参数。
- `models/segformer_cityscapes_cpu.yaml`：任务 6 冻结的 CPU 对照配置，保留第一版实验基线。
- `experiments/drivable_area_v1.yaml`：任务 6 冻结的数据引用、模型、512×512 输入、随机性、指标和性能测试口径。

后续训练、推理、评估和仿真配置应分别放入同名子目录，并在运行时将完整配置快照保存到对应的 `outputs/<run_id>/`。
