import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from configs.config import Config
from utils.utils import ensure_dir


def main():

    cfg = Config
    ensure_dir(cfg.MASK_OUTPUT_DIR)

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
    model.to(cfg.DEVICE)
    model.eval()

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

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = inputs.pixel_values.to(
            cfg.DEVICE
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

        road = (
            pred.squeeze().cpu().numpy() == 1
        ).astype(np.uint8)

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
