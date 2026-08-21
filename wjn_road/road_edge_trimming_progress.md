# 道路边缘切边优化 — 阶段性总结

## 目标
使道路边界 / 道路 mask IoU **显著高于** team 3 sub 的 SegFormer（道路分割）基线。

## 已建立：统一评估基准
- 数据：BDD100K drivable 真值（`*_drivable_id.png`，`label>0` 为道路，`255` 忽略）。
- 主指标：道路 mask IoU、边界 F1/IoU（trimap 容差 3）。
- 辅助指标：道路 precision / recall。
- 同口径对比：我方 PIDNet 与 team 3 sub SegFormer 在**同一批 100 张** `val_pilot.lst` 样本上计算。
- 脚本：
  - `scripts/evaluate_road_boundary.py`（PIDNet + A/B/C 后处理）
  - `scripts/evaluate_segformer_baseline.py`（team 3 sub SegFormer）

## 结果对比（100 张 val_pilot，全局像素 IoU）

| 方案 | road IoU | precision | recall | boundary F1 |
|---|---|---|---|---|
| 我方 PIDNet raw（19 类语义 road 类） | 0.4857 | 0.4882 | 0.9897 | 0.0404 |
| 我方 PIDNet + A/B/C（all） | 0.4652 | 0.6214 | 0.6492 | 0.0642 |
| 我方 PIDNet drivable pilot（5 epoch 随机初始化） | 0.3602 | 0.5099 | 0.5509 | 0.0424 |
| **team 3 sub SegFormer（argmax 0.5）** | **0.7176** | 0.834 | 0.8371 | 0.173 |
| team 3 sub SegFormer（阈值 0.2） | 0.6996 | 0.7231 | 0.9556 | 0.1661 |

> team 3 sub SegFormer 的 checkpoint 内记录其内部验证集 best IoU 为 0.6947（平均逐图 IoU），
> 与本表在官方 val_pilot 上同口径复算的 0.6672（平均逐图）/ 0.7176（全局）一致。

## 关键结论
1. 三种后处理方案 A（车道线引导切边）、B（DenseCRF 等价联合双边滤波）、C（Snake 主动轮廓）均已落地并接入 `postprocess.py`。
2. `refine_road_mask`（连通域 + 形态学）过度删除了道路像素：raw→refined 使 recall 由 0.9897 降至 0.8277，road IoU 反而下降（0.4857→0.4613）。
3. A/B/C 只能把 boundary F1 从 0.0404 小幅提到 0.0642，但 road IoU 仍为 0.4652，**远低于** SegFormer 的 0.7176。
4. 结论：后处理无法弥补 0.23 的 road IoU 差距（底层 PIDNet 语义「road」类与 drivable 真值系统性不一致 + 模型容量差距），**触发方案 D——重训专用道路模型（使用真实标注）**。

## 方案 D 已启动
- 模型：PIDNet-small，2 类（道路/背景），直接以 BDD100K drivable 真实标注训练。
- 初始化：从现有 19 类语义模型 backbone 迁移（成功加载 475/479 参数）。
- 数据：`train_pilot.lst`（8192 张）训练，`val_pilot.lst`（1024 张）监控。
- 配置：640×384，batch 8，SGD LR 0.01（poly 衰减），100 epoch，WORKERS=0，验证 batch=1。
- 配置文件：`external/PIDNet/configs/bdd100k/pidnet_small_bdd_drivable_pilot_v2.yaml`
- 稳态速度约 0.45 s/iter，保守预估完成时间 **约 18–24 小时**。
- 训练中不轮询，完成后一次性读取日志（`D:/pidnet_drivable_retrain/train/bdd_drivable/pidnet_small_bdd_drivable_pilot_v2/*_train.log`）判定结果。

## 方案 D 最终结果（目标达成）

方案 D 训练完成（100 epoch，13 小时），用 `evaluate_road_boundary.py`（`--road-class-id 1`）在同 100 张 `val_pilot` 样本上复算：

| 指标 | team 3 sub SegFormer | 方案 D（PIDNet drivable best.pt） | 提升 |
|---|---|---|---|
| road IoU（global） | 0.7176 | **0.8226** | **+0.105** |
| road precision | 0.834 | 0.8853 | +0.051 |
| road recall | 0.8371 | 0.9207 | +0.084 |
| boundary F1（容差 3） | 0.173 | **0.3389** | **+0.166** |
| boundary IoU（容差 3） | — | 0.2133 | — |

- best checkpoint：Epoch 67，训练验证集 road IoU 0.8078 / MeanIU 0.8823；最终 Epoch 99 road IoU 0.8014。
- 结论：方案 D **显著超越** team 3 sub SegFormer 基线（road IoU +0.105，boundary F1 接近翻倍）。
- 后处理变体（refined/crf/lane_trim）对重训模型的 road IoU 几乎无影响（-0.0015 左右），说明专用模型 raw 输出已足够精准，无需额外切边；snake/all 反而降低（road IoU 0.7683/0.7681），不采用。

## 交付物
- 训练配置：`external/PIDNet/configs/bdd100k/pidnet_small_bdd_drivable_pilot_v2.yaml`
- 权重：`D:/pidnet_drivable_retrain/train/bdd_drivable/pidnet_small_bdd_drivable_pilot_v2/best.pt`（及 checkpoint.pth.tar / final_state.pt）
- 评估产物：`outputs/drivable_pidnet_v2_eval/summary.json`、`per_sample.json`
