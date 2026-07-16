# 模型权重

模型权重文件不提交 Git。每个可用模型只在 `index.yaml` 中登记逻辑名称、版本、任务、外部存储位置、SHA256、训练配置和数据版本。

支持但默认忽略的常见格式包括 `.pt`、`.pth`、`.ckpt`、`.onnx`、`.engine` 和 `.safetensors`。

任务 5 已在 `index.yaml` 登记 `segformer-b0-cityscapes`，固定 Hugging Face revision 和权重 SHA-256。运行 `python scripts/download_segmentation_model.py` 后，下载文件位于 `checkpoints/segformer-b0-cityscapes/`；该目录内容全部保持本地，不进入 Git。
