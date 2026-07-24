import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============================================================
# 1. 配置
# ============================================================
IMAGE_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\train"
LABEL_DIR = r"E:\PythonProject1\data\raw\bdd100k\BDD100K.tar.gz\OpenDataLab___BDD100K\raw\BDD100K\BDD100K\Drivable Area\bdd100k_drivable_labels_trainval\bdd100k\labels\drivable\masks\train"
OUTPUT_DIR = "outputs/trained_model_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.05
NUM_WORKERS = 0
MAX_SAMPLES = 20000
IMAGE_SIZE = 512

CHECKPOINT_PATH = "outputs/trained_model_full/best_model_full.pt"
PATIENCE = 10

# 日志
log_filename = os.path.join(OUTPUT_DIR, f"training_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("🚀 SegFormer-B0 高速训练 (512x512, 20epoch, 2万张, batch=8)")
logger.info(f"设备: {DEVICE}, Max Epochs: {MAX_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
logger.info("="*60)

# ============================================================
# 2. 数据加载（手动预处理，避免 processor 重复 resize）
# ============================================================
# 归一化参数（ImageNet 标准值）
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class BDDSegDataset(torch.utils.data.Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        label = label.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        if self.augment and torch.rand(1) < 0.5:
            image = T.functional.hflip(image)
            label = T.functional.hflip(label)

        img_np = np.array(image, dtype=np.float32) / 255.0
        label_np = np.array(label)
        label_binary = (label_np >= 1).astype(np.int64)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        # 修复归一化
        img_t = (img_t - mean) / std
        label_t = torch.tensor(label_binary, dtype=torch.long)

        return {
            "pixel_values": img_t,
            "labels": label_t
        }

# ============================================================
# 3. 数据加载
# ============================================================
def check_data():
    if not os.path.exists(LABEL_DIR):
        logger.error(f"❌ 标注目录不存在: {LABEL_DIR}")
        return []
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    label_files = set(f for f in os.listdir(LABEL_DIR) if f.endswith('.png'))
    paired = []
    for img in img_files:
        label_name = img.replace('.jpg', '.png')
        if label_name in label_files:
            paired.append((os.path.join(IMAGE_DIR, img), os.path.join(LABEL_DIR, label_name)))
    logger.info(f"总配对: {len(paired)}")
    return paired

paired_data = check_data()
if not paired_data:
    sys.exit(1)

if MAX_SAMPLES and len(paired_data) > MAX_SAMPLES:
    paired_data = paired_data[:MAX_SAMPLES]
    logger.info(f"使用前 {MAX_SAMPLES} 张")

dataset = BDDSegDataset(paired_data, augment=True)
val_dataset = BDDSegDataset(paired_data, augment=False)
total = len(dataset)
val_size = int(VAL_SPLIT * total)
train_size = total - val_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_subset.dataset.augment = False

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

logger.info(f"训练: {len(train_subset)}, 验证: {len(val_subset)}")

# ============================================================
# 4. 模型加载
# ============================================================
logger.info("加载模型...")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True)
model.decode_head.classifier = nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    logger.info("✅ 加载已有权重")
else:
    logger.info("从头开始")

model.to(DEVICE)

# ============================================================
# 5. 训练
# ============================================================
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

best_loss = float("inf")
patience = 0
start_time = time.time()
epoch_times = []

logger.info("开始训练...")
for epoch in range(1, MAX_EPOCHS + 1):
    epoch_start = time.time()

    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Train {epoch}/{MAX_EPOCHS}"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        loss = model(pixel_values=pixel_values, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}/{MAX_EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step()

    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)
    avg_epoch_time = sum(epoch_times[-3:]) / min(len(epoch_times), 3)
    remaining_epochs = MAX_EPOCHS - epoch
    remaining_time = remaining_epochs * avg_epoch_time

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
        f"用时: {epoch_duration/60:.1f}min | "
        f"已用: {elapsed/60:.1f}min | "
        f"预计剩余: {remaining_time/60:.1f}min"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_opt.pt"))
        logger.info(f"   ✅ 最佳模型 (loss: {best_loss:.5f})")
    else:
        patience += 1
        if patience >= PATIENCE:
            logger.info(f"🛑 早停 (patience={PATIENCE})")
            break

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "segformer_bdd_opt_final.pt"))
logger.info(f"✅ 训练完成！总用时: {(time.time()-start_time)/60:.1f}min")
logger.info(f"🏆 最佳验证 loss: {best_loss:.5f}")