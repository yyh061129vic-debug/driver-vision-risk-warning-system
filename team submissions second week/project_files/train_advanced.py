import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 注意：不要设置 TRANSFORMERS_OFFLINE=1，否则无法下载

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
OUTPUT_DIR = "outputs/trained_model_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 6e-5
VAL_SPLIT = 0.05
NUM_WORKERS = 0
MAX_SAMPLES = None              # None = 全部数据
IMAGE_SIZE = 512

# 模型选择
MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
# 备选 B0（如果想换回 B0，注释掉上面这行，取消下面注释）
# MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

PATIENCE = 10

# 日志文件
log_filename = os.path.join(OUTPUT_DIR, f"training_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info(f"🚀 SegFormer 高级训练 (全量数据 + 增强)")
logger.info(f"模型: {MODEL_NAME}")
logger.info(f"设备: {DEVICE}, Max Epochs: {MAX_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
logger.info(f"数据: {'全部' if MAX_SAMPLES is None else MAX_SAMPLES} 张")
logger.info("="*60)

# ============================================================
# 2. 数据加载（增强版：翻转 + 旋转 + 颜色抖动）
# ============================================================
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

        if self.augment:
            # 随机翻转（50%）
            if torch.rand(1) < 0.5:
                image = T.functional.hflip(image)
                label = T.functional.hflip(label)
            # 随机旋转（-10° ~ 10°，30% 概率）
            if torch.rand(1) < 0.3:
                angle = (torch.rand(1) - 0.5) * 20
                image = T.functional.rotate(image, angle.item())
                label = T.functional.rotate(label, angle.item(), fill=0, interpolation=Image.NEAREST)
            # 颜色抖动（仅图像，30% 概率）
            if torch.rand(1) < 0.3:
                brightness = 0.8 + 0.4 * torch.rand(1)
                contrast = 0.8 + 0.4 * torch.rand(1)
                saturation = 0.8 + 0.4 * torch.rand(1)
                hue = 0.1 * (torch.rand(1) - 0.5)
                image = T.functional.adjust_brightness(image, brightness.item())
                image = T.functional.adjust_contrast(image, contrast.item())
                image = T.functional.adjust_saturation(image, saturation.item())
                image = T.functional.adjust_hue(image, hue.item())

        img_np = np.array(image, dtype=np.float32) / 255.0
        label_np = np.array(label)
        label_binary = (label_np >= 1).astype(np.int64)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        img_t = (img_t - mean) / std
        label_t = torch.tensor(label_binary, dtype=torch.long)

        return {"pixel_values": img_t, "labels": label_t}

# ============================================================
# 3. 加载数据
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
else:
    logger.info(f"✅ 使用全部 {len(paired_data)} 张")

dataset = BDDSegDataset(paired_data, augment=True)
val_dataset = BDDSegDataset(paired_data, augment=False)
total = len(dataset)
val_size = int(VAL_SPLIT * total)
train_size = total - val_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_subset.dataset.augment = False

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

logger.info(f"训练: {len(train_subset)}, 验证: {len(val_subset)}")

# ============================================================
# 4. 加载模型（正常下载，通过镜像加速）
# ============================================================
logger.info(f"加载模型: {MODEL_NAME}")
logger.info("首次运行会自动下载模型权重（约 1.2GB），请耐心等待...")
try:
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
except Exception as e:
    logger.error(f"下载失败: {e}")
    logger.info("尝试使用本地缓存（如果存在）...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, local_files_only=True)

model.decode_head.classifier = nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)
model.to(DEVICE)
logger.info("✅ 模型加载完成")

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

    # ---- 训练 ----
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

    # ---- 验证 ----
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

    # ---- 计时 ----
    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)
    avg_epoch_time = sum(epoch_times[-3:]) / min(len(epoch_times), 3)
    remaining_epochs = MAX_EPOCHS - epoch
    remaining_time = remaining_epochs * avg_epoch_time
    elapsed = time.time() - start_time

    # ---- 日志 ----
    logger.info(
        f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
        f"用时: {epoch_duration/60:.1f}min | "
        f"已用: {elapsed/60:.1f}min | "
        f"预计剩余: {remaining_time/60:.1f}min"
    )

    # ---- 保存最佳模型 ----
    if val_loss < best_loss:
        best_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_advanced.pt"))
        logger.info(f"   ✅ 保存最佳模型 (Val Loss: {best_loss:.5f})")
    else:
        patience += 1
        logger.info(f"   ⏳ 早停计数: {patience}/{PATIENCE}")

    # ---- 早停判断 ----
    if patience >= PATIENCE:
        logger.info(f"🛑 早停触发！连续 {PATIENCE} 轮 Val Loss 未改善，停止训练。")
        break

# ============================================================
# 6. 保存最终模型
# ============================================================
final_path = os.path.join(OUTPUT_DIR, "segformer_b2_advanced_final.pt")
torch.save(model.state_dict(), final_path)
logger.info(f"✅ 训练完成！总用时: {(time.time()-start_time)/60:.1f}min")
logger.info(f"🏆 最佳验证 Loss: {best_loss:.5f}")
logger.info(f"📁 模型保存在: {OUTPUT_DIR}")
logger.info(f"📋 日志文件: {log_filename}")