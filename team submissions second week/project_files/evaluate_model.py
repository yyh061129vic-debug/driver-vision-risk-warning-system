import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ============================================================
# 配置
# ============================================================
IMAGE_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\val"
GT_DIR = r"E:\PythonProject1\data\raw\bdd100k\BDD100K.tar.gz\OpenDataLab___BDD100K\raw\BDD100K\BDD100K\Drivable Area\bdd100k_drivable_labels_trainval\bdd100k\labels\drivable\masks\val"
MODEL_PATH = "outputs/trained_model_full/best_model_opt.pt"   # 新模型
# MODEL_PATH = "outputs/trained_model_full/best_model_full.pt"   # 旧模型（可切换）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 200

print(f"使用设备: {DEVICE}")

# 加载模型
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True
)
model.decode_head.classifier = torch.nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ 模型加载成功")

# 找图片
img_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:NUM_SAMPLES]
ious = []
accuracies = []

for img_name in tqdm(img_files, desc="评估中"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 真实标注
    label_name = img_name.replace('.jpg', '.png')
    label_path = os.path.join(GT_DIR, label_name)
    if not os.path.exists(label_path):
        continue
    gt = np.array(Image.open(label_path).resize((w, h), Image.NEAREST))
    gt_binary = (gt >= 1).astype(np.uint8)

    # 推理
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # IoU
    intersection = np.logical_and(pred, gt_binary).sum()
    union = np.logical_or(pred, gt_binary).sum()
    iou = intersection / union if union > 0 else 1.0
    ious.append(iou)

    # 像素准确率
    acc = (pred == gt_binary).mean()
    accuracies.append(acc)

# 结果
print("\n" + "="*60)
print("📊 评估结果")
print("="*60)
print(f"样本数: {len(ious)}")
print(f"平均 IoU:  {np.mean(ious):.4f}  ± {np.std(ious):.4f}")
print(f"平均像素准确率: {np.mean(accuracies):.4f}  ± {np.std(accuracies):.4f}")
print("="*60)