# 候选道路视觉数据集调研

调研日期：2026-07-16

调研范围：Cityscapes、BDD100K、Mapillary Vistas、Lost and Found、Fishyscapes、RoadAnomaly21、SegmentMeIfYouCan。
结论性质：下载前的技术与许可预审，不构成法律意见。

## 结论摘要

1. 第一版可行驶区域/多任务基线优先考虑 BDD100K，但必须先从官方下载流程保存并确认数据条款；工具代码的 BSD-3-Clause 许可证不能替代数据许可证。
2. 如果项目明确限定为非商业研究，可用 Cityscapes 建立稳定的可行驶区域与语义分割基线；其许可证禁止商业使用和数据再分发。
3. Lost and Found 适合真实小型道路障碍物验证，但同样限定非商业使用且禁止再分发。
4. Fishyscapes、RoadAnomaly21 和 SegmentMeIfYouCan 主要用于异常/障碍评估，不应作为第一版训练集。隐藏测试集和测试图片不得用于训练或阈值拟合。
5. Mapillary Vistas 更适合基线跑通后的跨地域泛化测试；本登记限定 v1.2，下载时仍需保存该版本展示的准确许可。

## 对比表

| 数据集 | 规模 | 类别与标注 | 许可结论 | 官方下载 | 建议用途 | 当前决策 |
| --- | --- | --- | --- | --- | --- | --- |
| Cityscapes | 5,000 张精标、20,000 张粗标、50 城市 | 30 个像素类别，常用评测 19 类；语义/实例/全景分割 | 仅非商业用途；禁止原始及可还原修改数据再分发；需注册并引用 | 注册登录后下载 `leftImg8bit` 和 `gtFine` | 可行驶区域与城市语义基线 | 条件采用；仅限非商业研究 |
| BDD100K | 100,000 段约 40 秒视频，超过 1,000 小时、1 亿帧；100,000 关键帧 | 10 类道路目标框、直接/可选可行驶区、车道属性；10,000 张 40 类密集分割 | 官方工具仓库为 BSD-3-Clause，但没有足够证据表明该许可证覆盖数据；下载条款必须另行保存确认 | 官方 Berkeley DeepDrive 门户按任务下载 | 首选真实道路多任务基线 | 许可阻塞，确认后优先 |
| Mapillary Vistas v1.2 | 25,000 张；18k/2k/5k | 66 个语义类别，37 个实例类别，全球多设备街景 | 官方论文说明图像源按 CC-BY-SA；下载时必须确认 v1.2 包的准确版本条款 | 官方 Vistas 研究数据门户注册下载 | 跨地域泛化与语义预训练 | 第二顺位，先跑通基础基线 |
| Lost and Found | 112 个双目序列，2,104 个标注帧；完整托管约 49.1 GB | 像素级障碍/自由空间，含双目、视差、相机和车辆信息 | 仅非商业用途；禁止数据及修改版再分发；不可还原的抽象模型可发布 | 研究机构数据卡；最小下载约 6 GB 左图加 37 MB 标签 | 真实小型道路障碍物验证 | 条件采用；按序列划分 |
| Fishyscapes | 公开 Lost & Found 验证集 100 张；标注包 729.3 kB；完整测试集服务器端保留 | 二值异常掩码；另有 Static/Web 隐藏评测 | 公开标注包为 CC BY 4.0，但底图继续受 Lost and Found/Cityscapes 限制 | 官网跳转 Zenodo 下载标注，底图另行合法取得 | 异常指标与阈值验证 | 仅评估，不用于训练 |
| RoadAnomaly21 | 100 张测试图加 10 张额外公开标注图；52.6 MB | anomaly / not anomaly / void 三类像素标签；异常可出现在全图 | Zenodo 标为 Other (Attribution)，包内包含逐图来源与许可；必须逐项复核 | SegmentMeIfYouCan 官网或 Zenodo AnomalyTrack | 跨域未知异常评估 | 完成逐图许可审计后仅评估 |
| SegmentMeIfYouCan（RoadObstacle21） | 327 张测试图加 30 张额外公开标注图；207.6 MB | obstacle / not obstacle / void，路面为 ROI；支持像素与组件级指标 | RoadObstacle21 官方 Zenodo 包为 CC BY 4.0；RoadAnomaly21 仍按其独立混合许可 | 官网或 Zenodo ObstacleTrack | 道路障碍像素级与组件级评测 | 推荐评测套件，不作训练语料 |

## 推荐的最小组合

### 路线 A：许可确认后的首选方案

- BDD100K：训练可行驶区域和已知道路目标基线。
- Lost and Found：真实小障碍物验证，不与 BDD100K 相邻帧/序列混合划分。
- Fishyscapes Lost & Found validation：异常像素指标和阈值验证。
- SegmentMeIfYouCan RoadObstacle21：最终像素级与组件级障碍评测。

### 路线 B：明确为非商业研究时的备用方案

- Cityscapes：可行驶区域和语义分割基线。
- Lost and Found：道路小障碍物验证。
- Fishyscapes 与 SegmentMeIfYouCan：隐藏/公开评测。
- Mapillary Vistas v1.2：泛化测试，不在第一轮训练中混入。

## 下载与入库门禁

下载前必须完成以下检查：

1. 保存下载当日的许可/条款快照，并记录版本、URL、日期和审查人。
2. 明确是否允许当前项目用途、模型发布、论文展示和商业化；不确定即禁止启用。
3. 估算磁盘占用，只下载任务所需包；原始数据写入 `data_raw/`，不得进入 Git。
4. 记录包名、字节数、SHA256、文件数量和解压脚本。
5. 先按序列/场景生成 train/val/test 清单，再进行任何预处理，防止相邻帧泄漏。
6. 完成类别、忽略区域、坐标系和标注格式映射；不得直接无规则混合。
7. 每个数据集随机输出至少 20 张标注叠加图并人工检查。

## 官方来源

- Cityscapes：[概览](https://www.cityscapes-dataset.com/)、[许可条款](https://www.cityscapes-dataset.com/license/)、[下载入口](https://www.cityscapes-dataset.com/downloads/)
- BDD100K：[官方下载](https://bdd-data.berkeley.edu/)、[官方工具仓库](https://github.com/bdd100k/bdd100k)、[标注格式](https://github.com/bdd100k/bdd100k/blob/master/doc/format.md)、[CVPR 2020 论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html)
- Mapillary Vistas：[ICCV 2017 论文](https://openaccess.thecvf.com/content_ICCV_2017/html/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.html)、[Mapillary 图像许可说明](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)
- Lost and Found：[研究机构数据卡与许可](https://huggingface.co/datasets/iis-esslingen/LostAndFoundDataset)、[IROS 2016 论文](https://arxiv.org/abs/1609.04653)
- Fishyscapes：[官方数据页](https://fishyscapes.com/dataset)、[100 张验证标注 Zenodo 记录](https://zenodo.org/records/6511227)
- RoadAnomaly21：[官方数据页](https://segmentmeifyoucan.com/datasets)、[AnomalyTrack Zenodo 记录](https://zenodo.org/records/5270237)
- SegmentMeIfYouCan：[官方数据页](https://segmentmeifyoucan.com/datasets)、[RoadObstacle21 Zenodo 记录](https://zenodo.org/records/5281633)、[论文](https://arxiv.org/abs/2104.14812)

详细机器可读字段见 [`metadata/dataset_registry.yaml`](../metadata/dataset_registry.yaml)。
