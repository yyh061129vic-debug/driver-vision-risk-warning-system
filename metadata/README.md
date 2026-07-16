# 数据元数据

此目录登记数据来源、许可状态、环境基线、版本、规模、校验值、类别映射、划分策略、限制和推荐用途。`dataset_registry.yaml` 保存机器可读调研结果，`licenses/` 用于保存下载当日的许可或条款快照，`environment/` 保存脱敏的开发环境快照。

每个正式数据版本应补充数据卡和完整性清单；敏感信息与访问令牌不得写入仓库。

完整对比与推荐见 [`docs/dataset-survey.md`](../docs/dataset-survey.md)。任何候选数据加入 `configs/data/datasets.yaml` 的 `enabled` 前，必须通过许可、存储、校验值和划分检查。
