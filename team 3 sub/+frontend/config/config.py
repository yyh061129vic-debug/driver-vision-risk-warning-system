import os
import torch


class Config:
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    EPOCHS = 20
    BATCH_SIZE = 2
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    # Model input size: (height, width)
    IMAGE_SIZE = (384, 640)
    NUM_CLASSES = 2

    # Dataset
    TRAIN_IMG = "./data/train/images"
    TRAIN_MASK = "./data/train/masks"
    VAL_IMG = "./data/val/images"
    VAL_MASK = "./data/val/masks"

    # SegFormer
    # Local HuggingFace-style model directory (config.json + model files).
    # If missing, the HuggingFace pretrained model below is used as fallback.
    MODEL_DIR = "./models/segformer-b0"

    # Fallback pretrained SegFormer used when MODEL_DIR / BEST_MODEL is absent.
    SEGFORMER_PRETRAINED = "nvidia/segformer-b0-finetuned-ade-512-512"

    # Output
    WEIGHTS_DIR = "./weights"
    BEST_MODEL = os.path.join(
        WEIGHTS_DIR,
        "best_segformer_road.pth",
    )

    # Prediction
    YOLO_WEIGHT = os.path.join(
        WEIGHTS_DIR,
        "yolo",
        "best.pt",
    )

    # 仓库 YOLOv11s-P2 权重为自定义 15 类:
    # 0 bus 1 traffic light 2 traffic sign 3 person 4 bike 5 truck
    # 6 motor 7 car 8 train 9 rider 10 animal 11 cone 12 lost_tire
    # 13 obstacle 14 vehicle
    # 车辆相关类别（用于从 road mask 中剔除）:
    YOLO_VEHICLE_CLASSES = (0, 5, 6, 7, 8, 14)

    TEST_DIR = "./data/test"
    MASK_OUTPUT_DIR = "./results/masks"

    # Web demo
    RESULT_DIR = "./results/web"

    # 视频处理（逐帧推理，FPS/分辨率保持原视频）:
    # 时长限制已彻底取消：None 表示处理完整视频（不截断）。
    MAX_VIDEO_SECONDS = None

    # 视频批处理推理：攒 N 帧一次性送入 GPU，显著提升吞吐（精度不变）。
    VIDEO_BATCH_SIZE = 8
