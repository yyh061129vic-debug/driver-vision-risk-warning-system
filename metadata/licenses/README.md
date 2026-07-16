# 数据许可快照

候选数据调研只记录了 2026-07-16 可从官方页面核验到的条款，不等同于下载授权，也不构成法律意见。

正式下载前，应在此目录为每个数据集保存下载当日看到的许可或使用条款，建议命名为：

```text
<dataset-id>/<YYYY-MM-DD>-license-or-terms.<html|pdf|txt>
<dataset-id>/<YYYY-MM-DD>-review.yaml
```

审查记录至少包含：版本、来源 URL、访问日期、允许用途、商业限制、再分发限制、署名要求、隐私要求、负责人和结论。BDD100K 必须特别确认数据条款，不能用工具仓库的 BSD-3-Clause 许可证替代；RoadAnomaly21 必须审查压缩包内逐图来源和许可。

任务 4 已新增 `lost-and-found-2026-07-16.md` 和 `road-obstacle-21-2026-07-16.md`。前者只批准本地非商业验证且禁止再分发，后者只启用 RoadObstacle21 的可视化与评测用途。

任务 5 已新增 `segformer-b0-cityscapes-2026-07-16.md`。该模型受 NVIDIA SegFormer Source Code License 约束，本项目仅将其用于本地非商业研究与评估，不得据此推定可商业部署。
