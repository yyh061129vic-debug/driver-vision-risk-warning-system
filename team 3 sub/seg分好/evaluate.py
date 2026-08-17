import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from datasets.road_dataset import RoadDataset
from models.segformer import build_segformer
from utils.metrics import (
    binary_iou,
    binary_dice,
    pixel_accuracy,
)
from utils.utils import load_checkpoint


def main():

    cfg = Config

    dataset = RoadDataset(
        cfg.VAL_IMG,
        cfg.VAL_MASK,
        size=cfg.IMAGE_SIZE,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    model = build_segformer(
        cfg.MODEL_DIR,
        num_labels=cfg.NUM_CLASSES,
        device=cfg.DEVICE,
    )

    load_checkpoint(
        model,
        cfg.BEST_MODEL,
        cfg.DEVICE,
    )

    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0

    with torch.no_grad():

        for images, masks in tqdm(
            loader,
            desc="Evaluating",
        ):

            images = images.to(cfg.DEVICE)
            masks = masks.to(cfg.DEVICE)

            outputs = model(
                pixel_values=images
            )

            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            pred = torch.argmax(
                logits,
                dim=1,
            )

            total_iou += binary_iou(
                pred,
                masks,
            )

            total_dice += binary_dice(
                pred,
                masks,
            )

            total_acc += pixel_accuracy(
                pred,
                masks,
            )

    n = len(loader)

    print("\nEvaluation")
    print("IoU:", total_iou / n)
    print("Dice:", total_dice / n)
    print("Pixel Accuracy:", total_acc / n)


if __name__ == "__main__":
    main()
