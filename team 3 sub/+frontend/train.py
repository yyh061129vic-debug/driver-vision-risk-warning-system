import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import Config
from dataset.road_dataset import RoadDataset
from losses.road_loss import RoadLoss
from models.segformer import build_segformer
from utils.metrics import binary_iou
from utils.utils import ensure_dir, save_checkpoint


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if "decode_head" not in name:
            param.requires_grad = False


def unfreeze_encoder(model):
    for param in model.parameters():
        param.requires_grad = True


def main():

    cfg = Config
    ensure_dir(cfg.WEIGHTS_DIR)

    print("Device:", cfg.DEVICE)

    train_dataset = RoadDataset(
        cfg.TRAIN_IMG,
        cfg.TRAIN_MASK,
        size=cfg.IMAGE_SIZE,
        augment=True,
    )

    val_dataset = RoadDataset(
        cfg.VAL_IMG,
        cfg.VAL_MASK,
        size=cfg.IMAGE_SIZE,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    print("加载 SegFormer")
    model = build_segformer(
        cfg.MODEL_DIR,
        num_labels=cfg.NUM_CLASSES,
        device=cfg.DEVICE,
    )

    print("冻结 Encoder")
    freeze_encoder(model)

    criterion = RoadLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    use_amp = cfg.DEVICE.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    best_iou = 0.0

    for epoch in range(cfg.EPOCHS):

        # 第6轮开始解冻 Encoder
        if epoch == 5:
            print("\n解冻 Encoder")
            unfreeze_encoder(model)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.LR * 0.3,
                weight_decay=cfg.WEIGHT_DECAY,
            )

        model.train()
        total_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}",
        )

        for images, masks in loop:

            images = images.to(
                cfg.DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                cfg.DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                outputs = model(
                    pixel_values=images
                )

                logits = F.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = criterion(
                    logits,
                    masks,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_postfix(
                loss=round(loss.item(), 4)
            )

        train_loss = (
            total_loss / len(train_loader)
        )

        model.eval()
        total_iou = 0.0

        with torch.no_grad():

            for images, masks in tqdm(
                val_loader,
                desc="Validation",
            ):

                images = images.to(
                    cfg.DEVICE
                )

                masks = masks.to(
                    cfg.DEVICE
                )

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

        val_iou = (
            total_iou / len(val_loader)
        )

        print(
            f"\nEpoch: {epoch + 1}",
            f"loss: {train_loss:.4f}",
            f"IoU: {val_iou:.4f}",
        )

        if val_iou > best_iou:
            best_iou = val_iou

            save_checkpoint(
                model,
                cfg.BEST_MODEL,
            )

            print(
                "保存最佳模型:",
                cfg.BEST_MODEL,
            )

    print("\n训练完成")
    print("Best IoU:", best_iou)


if __name__ == "__main__":
    main()
