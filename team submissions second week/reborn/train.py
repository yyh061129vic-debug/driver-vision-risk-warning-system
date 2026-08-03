import os
import torch

from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from dataset import RoadDataset
from loss import RoadLoss


# ======================
# 参数
# ======================

DEVICE = "cuda"

EPOCHS = 30
BATCH_SIZE = 4
LR = 5e-5


TRAIN_IMG = "./data/train/images"
TRAIN_MASK = "./data/train/masks"

VAL_IMG = "./data/val/images"
VAL_MASK = "./data/val/masks"


MODEL_DIR = "./models/segformer-b0"


# ======================
# 数据
# ======================

train_dataset = RoadDataset(
    TRAIN_IMG,
    TRAIN_MASK
)

val_dataset = RoadDataset(
    VAL_IMG,
    VAL_MASK
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)



# ======================
# 模型
# ======================

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True
)


model.to(DEVICE)


criterion = RoadLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


best_iou = 0



# ======================
# IoU
# ======================

def calc_iou(pred, mask):

    pred = pred == 1
    mask = mask == 1

    inter = (
        pred & mask
    ).sum().item()


    union = (
        pred | mask
    ).sum().item()


    if union == 0:
        return 0


    return inter / union



# ======================
# Train
# ======================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )


    for images, masks in loop:


        images = images.to(DEVICE)

        masks = masks.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(
            pixel_values=images
        )


        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        loss = criterion(
            logits,
            masks
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        loop.set_postfix(
            loss=round(loss.item(),4)
        )



    train_loss = total_loss / len(train_loader)



    # ======================
    # Validation
    # ======================

    model.eval()


    total_iou = 0


    with torch.no_grad():


        for images, masks in tqdm(
            val_loader,
            desc="Validation"
        ):


            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            outputs = model(
                pixel_values=images
            )


            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )


            pred = torch.argmax(
                logits,
                dim=1
            )


            total_iou += calc_iou(
                pred,
                masks
            )



    val_iou = total_iou / len(val_loader)



    print(
        f"\nEpoch {epoch+1}/{EPOCHS}",
        "loss:",
        round(train_loss,4),
        "IoU:",
        round(val_iou,4)
    )



    if val_iou > best_iou:

        best_iou = val_iou


        torch.save(
            model.state_dict(),
            "best_segformer_b0.pth"
        )


        print("保存最佳模型")



print(
    "训练完成 best IoU:",
    best_iou
)