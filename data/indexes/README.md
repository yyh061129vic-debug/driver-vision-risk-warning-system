# 数据索引

样本级索引采用 JSON Lines：每行一个样本对象，并符合 `manifest.schema.json`。建议分别建立 `train.jsonl`、`val.jsonl` 和 `test.jsonl`，同时记录数据集、序列、场景、帧号、时间戳及标注路径。实验级冻结划分可使用 YAML，以便同时记录分区用途、序列隔离和变更规则。

索引只保存相对路径和元数据，不保存图像、视频、掩码或点云本体。

`task4_samples.jsonl` 登记任务 4 的 20 个本地可视化样例，其中 Lost and Found 与 RoadObstacle21 各 10 个。索引中的原图和掩码路径指向 Git 忽略的数据目录。

`drivable_area_v1_split.yaml` 是任务 6 的冻结划分清单：6 张 Lost and Found 训练帧只做开发冒烟检查；4 张 Lost and Found 测试帧与 RoadObstacle21 全部 30 张公开验证图构成 34 张留出评测集。划分按序列或场景隔离，不用于本地训练或阈值拟合。
