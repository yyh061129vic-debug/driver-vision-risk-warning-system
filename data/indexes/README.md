# 数据索引

索引采用 JSON Lines：每行一个样本对象，并符合 `manifest.schema.json`。建议分别建立 `train.jsonl`、`val.jsonl` 和 `test.jsonl`，同时记录数据集、序列、场景、帧号、时间戳及标注路径。

索引只保存相对路径和元数据，不保存图像、视频、掩码或点云本体。
