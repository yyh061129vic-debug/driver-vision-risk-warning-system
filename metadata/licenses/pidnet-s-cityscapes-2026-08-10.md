# PIDNet-S Cityscapes ONNX 资产许可与用途记录

- 记录日期：2026-08-10
- 模型标识：`pidnet-s-cityscapes-onnx-float`
- 资产来源：
  - 源实现仓库：`https://github.com/XuJiacong/PIDNet`
  - 导出资产卡片：`https://huggingface.co/qualcomm/PidNet`
  - 固定下载地址：`https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/pidnet/releases/v0.59.0/pidnet-onnx-float.zip`

## 已核对信息

1. `XuJiacong/PIDNet` 仓库 `LICENSE` 为 MIT License。
2. Qualcomm `PidNet` 模型卡说明该资产基于原始 `PidNet` 实现导出，页面元数据标注 `license: other`。
3. 模型卡公开提供 ONNX/TFLite/QNN 预导出资产下载链接，但未在卡片正文中给出足够明确的再分发授权表述。

## 本项目的使用结论

- 当前仓库只将该 `PIDNet-S` ONNX 资产用于：
  - 本地研究
  - 本地评估
  - 本地功能接入验证
- 不将下载得到的模型文件提交到 Git。
- 不据此推定可商业部署、可车规部署或可再分发。

## 备注

若后续要把该导出资产复制到其他仓库、交付给第三方或用于正式部署，应再次核验 Qualcomm AI Hub 对该导出资产的完整许可条款。
