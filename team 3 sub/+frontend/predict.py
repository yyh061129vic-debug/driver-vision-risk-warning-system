import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from config.config import Config
from utils.utils import ensure_dir


def load_segformer(cfg=None):
    """加载 SegFormer 处理器与模型。

    优先级:
      1. 本地 models/segformer-b0 + weights/best_segformer_road.pth（BDD 训练权重，road = class 1）
      2. HuggingFace 预训练权重（兜底，road 类别从 id2label 动态解析）

    返回 (processor, model, road_class, device, source)
    """
    cfg = cfg or Config

    device = cfg.DEVICE
    road_class = 1
    source = "local"

    local_model_ready = (
        os.path.isdir(cfg.MODEL_DIR)
        and os.path.exists(cfg.BEST_MODEL)
    )

    if local_model_ready:
        processor = SegformerImageProcessor.from_pretrained(
            cfg.MODEL_DIR,
            size={
                "height": cfg.IMAGE_SIZE[0],
                "width": cfg.IMAGE_SIZE[1],
            },
        )

        model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.MODEL_DIR,
            num_labels=cfg.NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

        checkpoint = torch.load(
            cfg.BEST_MODEL,
            map_location="cpu",
        )

        model.load_state_dict(checkpoint)
    else:
        source = "pretrained-fallback"

        processor = SegformerImageProcessor.from_pretrained(
            cfg.SEGFORMER_PRETRAINED,
            size={
                "height": cfg.IMAGE_SIZE[0],
                "width": cfg.IMAGE_SIZE[1],
            },
        )

        model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.SEGFORMER_PRETRAINED,
        )

        road_class = _find_road_class(model)

    model.to(device)
    model.eval()

    return processor, model, road_class, device, source


def _find_road_class(model):
    """从 id2label 中解析 road 类别索引（默认 ADE20K road = 6）。"""
    id2label = getattr(model.config, "id2label", {})

    for idx, label in id2label.items():
        if label.lower() == "road":
            return int(idx)

    return 6


def predict_road(
    processor,
    model,
    image,
    device,
    road_class=1,
):
    """对单张 PIL 图像执行道路分割，返回二值 mask（uint8, 0/1）。"""
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs.pixel_values.to(
        device
    )

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values
        )

    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred = torch.argmax(
        logits,
        dim=1,
    )

    mask = pred.squeeze().cpu().numpy()

    return (mask == road_class).astype(np.uint8)


def main():

    cfg = Config
    ensure_dir(cfg.MASK_OUTPUT_DIR)

    processor, model, road_class, device, source = load_segformer(cfg)

    print("模型来源:", source)

    images = sorted(
        [
            f for f in os.listdir(cfg.TEST_DIR)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    print("发现图片:", len(images))

    for index, name in enumerate(images):

        path = os.path.join(
            cfg.TEST_DIR,
            name,
        )

        image = Image.open(
            path
        ).convert("RGB")

        road = predict_road(
            processor,
            model,
            image,
            device,
            road_class,
        )

        save_path = os.path.join(
            cfg.MASK_OUTPUT_DIR,
            f"mask_{index}.png",
        )

        cv2.imwrite(
            save_path,
            road * 255,
        )

        print(
            f"[{index + 1}/{len(images)}] {name}"
        )

    print("预测完成")


if __name__ == "__main__":
    main()
