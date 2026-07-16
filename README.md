# 车载视觉道路障碍物与未知风险预警系统

本仓库用于建设“可行驶区域分割 → 已知障碍物检测 → 未知风险发现 → 空间冲突判断 → 风险预警”的可复现工程链路。

当前已完成第一周启动任务 1（仓库与目录）、任务 2（候选数据调研）、任务 3（本地开发环境基线）、任务 4（20 张道路/异常样例可视化）、任务 5（可行驶区域分割 Demo）和任务 6（冻结第一版实验方案）。V1 已固定数据划分、SegFormer revision、512×512 实际模型输入、随机种子和三项主要质量指标；需求尚未给出的数值验收线、目标 GPU 和风险阈值仍保持待确认。

## 目录约定

```text
.
├─ src/driver_vision_risk/  # Python 源码，按职责拆分
├─ configs/                # 路径、系统和实验配置
├─ data/                   # 可版本化的数据样本索引与格式定义
├─ data_raw/               # 原始数据，只读，不提交数据本体
├─ data_processed/         # 可重建的处理结果，不提交数据本体
├─ metadata/               # 数据来源、许可、校验与数据卡登记
├─ checkpoints/            # 权重索引；权重文件不进入 Git
├─ outputs/                # 日志、预测、可视化和报告；产物不进入 Git
├─ scripts/                # 下载、转换、校验、评估和仿真辅助脚本
├─ tests/                  # 单元与工程结构测试
├─ team_submissions/       # 组员个人提交区与协作模板
└─ docs/                   # 架构、接口和开发说明
```

代码、配置、数据索引、权重和运行输出分别由 `src/`、`configs/`、`data/indexes/`、`checkpoints/` 和 `outputs/` 管理。原始数据与处理数据分别位于 `data_raw/` 和 `data_processed/`。数据本体、模型权重和运行产物均通过各目录的忽略规则排除，仅提交索引、元数据与说明。

组员协作采用 `team_submissions/<GitHub用户名>/` 独立目录和个人分支；详细约定见 `team_submissions/README.md`。目录不能代替 GitHub 权限，没有写权限的组员应通过 Fork 和 Pull Request 提交。

## 快速验证

环境要求：Python 3.10 或更高版本。

```bash
python scripts/verify_layout.py
python scripts/validate_dataset_registry.py
python scripts/validate_environment_baseline.py
python scripts/validate_task4_visualizations.py
python scripts/validate_segmentation_demo.py
python scripts/validate_experiment_plan.py
python -m compileall -q src scripts tests
```

安装开发依赖后可运行：

```bash
python -m pip install -e ".[dev]"
pytest
driver-vision-risk --show-layout
```

任务 5 的 RTX 4070 / CUDA 13.0 环境安装与运行：

```powershell
python -m pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,inference]"
python scripts/download_segmentation_model.py
python scripts/run_segmentation_demo.py --input <图像或视频路径> --output outputs/task5_segmentation_demo/gpu
python scripts/validate_segmentation_demo.py
```

如需复现任务 6 冻结时的 CPU 对照，运行 Demo 时额外传入 `--config configs/models/segformer_cityscapes_cpu.yaml`；不要覆盖冻结的 CPU 实验记录。

## 配置原则

- 路径集中在 `configs/paths.yaml`，核心逻辑不硬编码本地绝对路径。
- 系统状态和待确认项登记在 `configs/system.yaml`；尚未确认的硬件与阈值保持 `null`。
- 数据候选与启用项分开登记；对比报告见 `docs/dataset-survey.md`，下载前必须保存当日许可快照并完成用途审批。
- 本地环境实测结果见 `docs/environment-baseline.md`；开发机快照不能替代尚待确认的目标 GPU 配置。
- 任务 4 采用 Lost and Found 与 RoadObstacle21；20 张叠加图的规则、位置与许可边界见 `docs/sample-visualization.md`。
- 任务 5 的模型版本、运行产物、实测指标和限制见 `docs/segmentation-demo.md`。
- 任务 6 的冻结数据划分、输入尺寸、指标口径和变更规则见 `docs/experiment-plan-v1.md`。
- 每个权重只在 `checkpoints/index.yaml` 登记版本、来源和校验信息。
- 每次实验应在独立的 `outputs/<run_id>/` 下保存配置快照、版本、随机种子、日志和结果。

## 当前状态

- [x] 建立仓库与标准目录
- [x] 分离代码、配置、数据索引、权重索引和输出
- [x] 提供大文件忽略规则和结构验证脚本
- [x] 完成候选数据调研与下载前许可预审
- [x] 记录本地开发环境基线（缺失组件明确标记为未安装）
- [x] 输出 20 张道路与异常样例可视化
- [x] 跑通可行驶区域分割 Demo
- [x] 冻结第一版实验方案
