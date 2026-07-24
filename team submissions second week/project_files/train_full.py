import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2

# ============================================================
# 1. 配置
# ============================================================
IMAGE_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\train"
LABEL_DIR = r"E:\PythonProject1\data\raw\bdd100k\BDD100K.tar.gz\OpenDataLab___BDD100K\raw\BDD100K\BDD100K\Drivable Area\bdd100k_drivable_labels_trainval\bdd100k\labels\drivable\masks\train"

OUTPUT_DIR = "outputs/trained_model_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 8                      # 根据显存调整（RTX 3060 6GB 可尝试 8）
LEARNING_RATE = 1e-4
MAX_SAMPLES = None                  # None 表示全部数据
VAL_SPLIT = 0.05                    # 从训练集分出 5% 作为验证集
NUM_WORKERS = 4

# ============================================================
# 2. 日志
# ============================================================
log_filename = os.path.join(OUTPUT_DIR, f"training_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("🚀 SegFormer-B0 全量微调训练 (二分类: 道路/非道路)")
logger.info(f"设备: {DEVICE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
logger.info(f"图片目录: {IMAGE_DIR}")
logger.info(f"标注目录: {LABEL_DIR}")
logger.info(f"验证集比例: {VAL_SPLIT*100:.1f}%")
logger.info("=" * 60)

# ============================================================
# 3. 检查数据
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

    logger.info(f"图片: {len(img_files)} 张, 标注: {len(label_files)} 个, 配对: {len(paired)} 对")
    return paired

paired_data = check_data()
if len(paired_data) == 0:
    logger.error("❌ 无配对数据，请检查标注路径")
    sys.exit(1)

if MAX_SAMPLES and len(paired_data) > MAX_SAMPLES:
    paired_data = paired_data[:MAX_SAMPLES]
    logger.info(f"使用前 {MAX_SAMPLES} 对数据训练")
else:
    logger.info(f"✅ 使用全部 {len(paired_data)} 对数据训练")

# ============================================================
# 4. 数据集
# ============================================================
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True)

class BDDSegDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        image = Image.open(img_path).convert("RGB").resize((1024, 1024))
        label = Image.open(label_path).resize((1024, 1024), Image.NEAREST)
        label_np = np.array(label)
        label_binary = (label_np >= 1).astype(np.int64)
        inputs = processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label_binary, dtype=torch.long)
        }

dataset = BDDSegDataset(paired_data)
total_size = len(dataset)
val_size = int(VAL_SPLIT * total_size)
train_size = total_size - val_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
logger.info(f"训练集: {train_size} 张, 验证集: {val_size} 张")

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# 5. 预估训练时间（warm-up）
# ============================================================
logger.info("⏳ 估算训练时间...")
warmup_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
dummy_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True)
dummy_model.decode_head.classifier = nn.Conv2d(dummy_model.config.decoder_hidden_size, 2, kernel_size=1)
dummy_model.to(DEVICE)
dummy_model.eval()

start_time = time.time()
with torch.no_grad():
    for i, batch in enumerate(warmup_loader):
        if i >= 5:  # 只跑5批估算
            break
        pixel_values = batch["pixel_values"].to(DEVICE)
        _ = dummy_model(pixel_values=pixel_values)
avg_batch_time = (time.time() - start_time) / 5
total_batches = len(train_loader)
estimated_epoch_time = avg_batch_time * total_batches
estimated_total_time = estimated_epoch_time * EPOCHS
logger.info(f"✅ 平均每批耗时: {avg_batch_time:.2f}s, 预计每轮: {estimated_epoch_time/60:.1f}分钟, 总训练: {estimated_total_time/3600:.1f}小时")
# 释放 dummy 模型以节省显存
del dummy_model
torch.cuda.empty_cache()

# ============================================================
# 6. 加载正式模型
# ============================================================
logger.info("加载预训练模型...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    local_files_only=True
)
model.decode_head.classifier = nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)
model.to(DEVICE)
logger.info("✅ 模型加载并修改分类头（2 类）")

# ============================================================
# 7. 训练
# ============================================================
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float("inf")
logger.info("开始训练...")

for epoch in range(1, EPOCHS + 1):
    # 训练
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step()
    logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_full.pt"))
        logger.info(f"   ✅ 保存最佳模型 (val_loss: {best_val_loss:.6f})")

# ============================================================
# 8. 保存最终模型
# ============================================================
final_path = os.path.join(OUTPUT_DIR, "segformer_bdd_full_final.pt")
torch.save(model.state_dict(), final_path)
logger.info(f"✅ 训练完成！最终模型保存在: {final_path}")
logger.info(f"📋 日志: {log_filename}")

# ============================================================
# 9. 自动测试评估
# ============================================================
logger.info("\n" + "="*60)
logger.info("🧪 开始自动测试评估...")
logger.info("="*60)

# 加载最佳模型进行测试
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model_full.pt"), map_location=DEVICE))
model.eval()

ious = []
accuracies = []
test_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)  # 逐张测试方便计算IoU

for idx, batch in enumerate(tqdm(test_loader, desc="测试进度")):
    pixel_values = batch["pixel_values"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)  # (1, H, W)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    gt = labels.squeeze(0).cpu().numpy()
    # 统一尺寸（logits 是 1/4 尺寸，已上采样到原图大小？实际上需要 resize）
    # 由于验证集已经 resize 到 1024x1024，直接比较
    # 但 logits 尺寸是 (1, 2, H/4, W/4)，需要上采样到原尺寸
    # 我们在推理中直接上采样
    # 更简单：用 inference 方式得到 pred_mask，但为了准确，我们手动上采样
    # 实际模型输出 logits 尺寸为 (1, 2, 256, 256)
    # 我们使用 interpolate 上采样到 1024x1024
    logits_up = torch.nn.functional.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
    pred_up = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy()
    gt = labels.squeeze(0).cpu().numpy()
    intersection = np.logical_and(pred_up, gt).sum()
    union = np.logical_or(pred_up, gt).sum()
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    ious.append(iou)
    acc = (pred_up == gt).mean()
    accuracies.append(acc)

logger.info(f"✅ 测试完成，样本数: {len(ious)}")
logger.info(f"📊 平均 IoU:  {np.mean(ious):.4f}  ± {np.std(ious):.4f}")
logger.info(f"📊 平均像素准确率: {np.mean(accuracies):.4f}  ± {np.std(accuracies):.4f}")
logger.info("="*60)
logger.info("🎉 全量训练与测试全部完成！")