import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from transformers import SegformerForSemanticSegmentation

from tqdm import tqdm

from dataset import RoadDataset
from loss import RoadLoss



# ======================
# 参数
# ======================

DEVICE="cuda"


EPOCHS=20

BATCH_SIZE=2


LR=5e-5



TRAIN_IMG="./data/train/images"
TRAIN_MASK="./data/train/masks"


VAL_IMG="./data/val/images"
VAL_MASK="./data/val/masks"



MODEL_DIR="./models/segformer-b0"



SAVE_PATH="best_segformer_road.pth"



# ======================
# Dataset
# ======================


train_dataset=RoadDataset(
    TRAIN_IMG,
    TRAIN_MASK,
    augment=True
)



val_dataset=RoadDataset(
    VAL_IMG,
    VAL_MASK,
    augment=False
)




train_loader=DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2,

    pin_memory=True

)



val_loader=DataLoader(

    val_dataset,

    batch_size=1,

    shuffle=False,

    num_workers=2

)



# ======================
# Model
# ======================


print("加载 SegFormer")



model=SegformerForSemanticSegmentation.from_pretrained(

    MODEL_DIR,

    num_labels=2,

    ignore_mismatched_sizes=True

)



model.to(DEVICE)



# ======================
# Encoder 冻结控制
# ======================


def freeze_encoder():

    for name,param in model.named_parameters():

        if "decode_head" not in name:

            param.requires_grad=False



def unfreeze_encoder():

    for param in model.parameters():

        param.requires_grad=True




print("冻结 Encoder")

freeze_encoder()



# ======================
# Loss
# ======================


criterion=RoadLoss()



# ======================
# Optimizer
# ======================


optimizer=torch.optim.AdamW(

    filter(
        lambda p:p.requires_grad,
        model.parameters()
    ),

    lr=LR,

    weight_decay=1e-4

)



# AMP

scaler=torch.cuda.amp.GradScaler()



best_iou=0



# ======================
# IoU
# ======================


def calc_iou(pred,mask):


    pred=pred==1

    mask=mask==1



    inter=(pred & mask).sum().item()



    union=(pred | mask).sum().item()



    if union==0:

        return 0



    return inter/union




# ======================
# Train
# ======================


for epoch in range(EPOCHS):


    # 第6轮解冻

    if epoch==5:

        print("\n解冻 Encoder")

        unfreeze_encoder()


        optimizer=torch.optim.AdamW(

            model.parameters(),

            lr=LR*0.3,

            weight_decay=1e-4

        )



    model.train()


    total_loss=0



    loop=tqdm(

        train_loader,

        desc=f"Epoch {epoch+1}/{EPOCHS}"

    )



    for images,masks in loop:


        images=images.to(
            DEVICE,
            non_blocking=True
        )


        masks=masks.to(
            DEVICE,
            non_blocking=True
        )



        optimizer.zero_grad()



        with torch.cuda.amp.autocast():


            outputs=model(

                pixel_values=images

            )



            logits=F.interpolate(

                outputs.logits,

                size=masks.shape[-2:],

                mode="bilinear",

                align_corners=False

            )



            loss=criterion(

                logits,

                masks

            )



        scaler.scale(loss).backward()


        scaler.step(
            optimizer
        )


        scaler.update()



        total_loss+=loss.item()



        loop.set_postfix(

            loss=round(loss.item(),4)

        )



    train_loss=total_loss/len(train_loader)




    # ======================
    # Validation
    # ======================


    model.eval()


    total_iou=0



    with torch.no_grad():


        for images,masks in tqdm(

            val_loader,

            desc="Validation"

        ):


            images=images.to(
                DEVICE
            )


            masks=masks.to(
                DEVICE
            )



            outputs=model(

                pixel_values=images

            )



            logits=F.interpolate(

                outputs.logits,

                size=masks.shape[-2:],

                mode="bilinear",

                align_corners=False

            )



            pred=torch.argmax(

                logits,

                dim=1

            )



            total_iou+=calc_iou(

                pred,

                masks

            )




    val_iou=total_iou/len(val_loader)



    print(

        "\nEpoch:",epoch+1,

        "loss:",round(train_loss,4),

        "IoU:",round(val_iou,4)

    )




    if val_iou>best_iou:


        best_iou=val_iou



        torch.save(

            model.state_dict(),

            SAVE_PATH

        )


        print("保存最佳模型")




print("\n训练完成")

print(
    "Best IoU:",
    best_iou
)