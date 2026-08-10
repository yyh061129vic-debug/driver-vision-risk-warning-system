import os
import torch
import cv2
import numpy as np

from PIL import Image

from ultralytics import YOLO

from transformers import SegformerForSemanticSegmentation
from transformers import SegformerImageProcessor



# ==========================
# 参数
# ==========================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# YOLO权重

YOLO_WEIGHT = "./weights/yolo.pt"


# SegFormer目录

SEGFORMER_DIR = "./models/segformer-b0"


# Decoder训练后的权重

SEGFORMER_WEIGHT = "./best_segformer_decoder.pth"



# 输入图片

INPUT_DIR="./data/test/images"


# 输出mask

OUTPUT_DIR="./results/masks"



os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==========================
# 加载YOLO
# ==========================


print("加载YOLO...")


yolo = YOLO(
    YOLO_WEIGHT
)



# ==========================
# 加载SegFormer
# ==========================


print("加载SegFormer...")


processor = SegformerImageProcessor.from_pretrained(
    SEGFORMER_DIR
)



segformer = SegformerForSemanticSegmentation.from_pretrained(
    SEGFORMER_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True
)



checkpoint=torch.load(
    SEGFORMER_WEIGHT,
    map_location="cpu"
)



segformer.load_state_dict(
    checkpoint
)



segformer.to(
    DEVICE
)


segformer.eval()



print("模型加载完成")





# ==========================
# SegFormer预测
# ==========================


def predict_road(image):


    inputs=processor(
        images=image,
        return_tensors="pt"
    )


    pixel_values=inputs.pixel_values.to(
        DEVICE
    )



    with torch.no_grad():


        outputs=segformer(
            pixel_values=pixel_values
        )



    logits=outputs.logits



    logits=torch.nn.functional.interpolate(

        logits,

        size=image.size[::-1],

        mode="bilinear",

        align_corners=False

    )



    pred=torch.argmax(
        logits,
        dim=1
    )


    mask=pred.squeeze().cpu().numpy()



    # 道路类别

    road=(mask==1).astype(
        np.uint8
    )


    return road






# ==========================
# YOLO车辆检测
# ==========================


def detect_vehicle(
    image
):


    results=yolo(
        image,
        verbose=False
    )



    vehicle_mask=np.zeros(

        image.shape[:2],

        dtype=np.uint8

    )



    for r in results:


        boxes=r.boxes



        for box in boxes:


            cls=int(
                box.cls[0]
            )


            conf=float(
                box.conf[0]
            )



            # COCO车辆类别
            # car=2
            # motorcycle=3
            # bus=5
            # truck=7


            if cls in [
                2,
                3,
                5,
                7
            ] and conf>0.3:



                x1,y1,x2,y2=map(
                    int,
                    box.xyxy[0]
                )


                vehicle_mask[
                    y1:y2,
                    x1:x2
                ]=1



    return vehicle_mask





# ==========================
# 融合
# ==========================


def fusion(
    road_mask,
    vehicle_mask
):


    # 删除车辆区域

    road_mask[
        vehicle_mask==1
    ]=0



    return road_mask






# ==========================
# 主程序
# ==========================


images=[

    f for f in os.listdir(INPUT_DIR)

    if f.lower().endswith(
        (
            ".jpg",
            ".png",
            ".jpeg"
        )
    )

]



for name in images:


    path=os.path.join(
        INPUT_DIR,
        name
    )



    image=Image.open(
        path
    ).convert(
        "RGB"
    )



    img_np=np.array(
        image
    )



    print(
        "预测:",
        name
    )



    # SegFormer道路

    road_mask=predict_road(
        image
    )



    # YOLO车辆

    vehicle_mask=detect_vehicle(
        img_np
    )



    # 融合

    final_mask=fusion(
        road_mask,
        vehicle_mask
    )



    save_path=os.path.join(

        OUTPUT_DIR,

        os.path.splitext(name)[0]+".png"

    )



    cv2.imwrite(

        save_path,

        final_mask*255

    )



print("全部完成")