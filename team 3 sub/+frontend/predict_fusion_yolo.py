import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from configs.config import Config
from utils.utils import ensure_dir


def refine_mask(mask, kernel_size=5, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    refined = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return refined


def main():

    cfg = Config
    ensure_dir(cfg.MASK_OUTPUT_DIR)

    print("加载YOLO...")
    yolo = YOLO(cfg.YOLO_WEIGHT)

    print("加载SegFormer...")
    processor = SegformerImageProcessor.from_pretrained(
        cfg.MODEL_DIR,
        size={
            "height": cfg.IMAGE_SIZE[0],
            "width": cfg.IMAGE_SIZE[1],
        },
    )

    segformer = SegformerForSemanticSegmentation.from_pretrained(
        cfg.MODEL_DIR,
        num_labels=cfg.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(
        cfg.BEST_MODEL,
        map_location="cpu",
    )

    segformer.load_state_dict(checkpoint)
    segformer.to(cfg.DEVICE)
    segformer.eval()

    def predict_road(image):
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = inputs.pixel_values.to(
            cfg.DEVICE
        )

        with torch.no_grad():
            outputs = segformer(
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

        return (mask == 1).astype(np.uint8)

    def detect_vehicle(image):
        results = yolo(
            image,
            verbose=False,
        )

        vehicle_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [2, 3, 5, 7] and conf > 0.3:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0],
                    )

                    vehicle_mask[
                        y1:y2,
                        x1:x2
                    ] = 1

        return vehicle_mask

    def fusion(road_mask, vehicle_mask):
        road_mask[
            vehicle_mask == 1
        ] = 0

        return road_mask

    images = sorted(
        [
            f for f in os.listdir(cfg.TEST_DIR)
            if f.lower().endswith(
                (".jpg", ".png", ".jpeg")
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

        img_np = np.array(image)

        print("预测:", name)

        road_mask = predict_road(image)

        road_mask = refine_mask(road_mask, kernel_size=5, iterations=1)

        vehicle_mask = detect_vehicle(img_np)
        final_mask = fusion(road_mask, vehicle_mask)

        save_path = os.path.join(
            cfg.MASK_OUTPUT_DIR,
            f"fusion_{index}.png",
        )

        cv2.imwrite(
            save_path,
            final_mask * 255,
        )

    print("全部完成")


if __name__ == "__main__":
    main()