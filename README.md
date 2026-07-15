# 车载视觉道路障碍物与未知风险预警系统

本仓库用于建设“可行驶区域分割 → 已知障碍物检测 → 未知风险发现 → 空间冲突判断 → 风险预警”的可复现工程链路。

当前里程碑仅完成第一周启动任务 1：建立代码仓库和标准目录。模型、阈值、数据集与仿真业务逻辑尚未实现，避免在摄像头参数、已知类别清单、风险阈值和目标 GPU 未确认前固化错误假设。

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
└─ docs/                   # 架构、接口和开发说明
```

代码、配置、数据索引、权重和运行输出分别由 `src/`、`configs/`、`data/indexes/`、`checkpoints/` 和 `outputs/` 管理。原始数据与处理数据分别位于 `data_raw/` 和 `data_processed/`。数据本体、模型权重和运行产物均通过各目录的忽略规则排除，仅提交索引、元数据与说明。

## 快速验证

环境要求：Python 3.10 或更高版本。

```bash
python scripts/verify_layout.py
python -m compileall -q src scripts tests
```

安装开发依赖后可运行：

```bash
python -m pip install -e ".[dev]"
pytest
driver-vision-risk --show-layout
```

## 配置原则

- 路径集中在 `configs/paths.yaml`，核心逻辑不硬编码本地绝对路径。
- 系统状态和待确认项登记在 `configs/system.yaml`；尚未确认的硬件与阈值保持 `null`。
- 数据候选与启用项分开登记；下载前必须完成许可核验。
- 每个权重只在 `checkpoints/index.yaml` 登记版本、来源和校验信息。
- 每次实验应在独立的 `outputs/<run_id>/` 下保存配置快照、版本、随机种子、日志和结果。

## 当前状态

- [x] 建立仓库与标准目录
- [x] 分离代码、配置、数据索引、权重索引和输出
- [x] 提供大文件忽略规则和结构验证脚本
- [ ] 完成候选数据调研与许可核验
- [ ] 记录目标环境基线
- [ ] 跑通可行驶区域分割 Demo
- [ ] 冻结第一版实验方案
