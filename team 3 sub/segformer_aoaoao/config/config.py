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
    MODEL_DIR = "./models/segformer-b0"

    # Output
    WEIGHTS_DIR = "./weights"
    BEST_MODEL = os.path.join(
        WEIGHTS_DIR,
        "best_segformer_road.pth",
    )

    # Prediction
    YOLO_WEIGHT = "/home/epfl/chat/yolo_warning/weights/best.pt"
    TEST_DIR = "/home/epfl/chat/test/"
    MASK_OUTPUT_DIR = "./results/masks"
