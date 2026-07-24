import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 配置
IMAGE_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\val"
GT_DIR = r"E:\PythonProject1\data\raw\bdd100k\BDD100K.tar.gz\OpenDataLab___BDD100K\raw\BDD100K\BDD100K\Drivable Area\bdd100k_drivable_labels_trainval\bdd100k\labels\drivable\masks\val"

OLD_MODEL = "outputs/trained_model_full/best_model_full.pt"
NEW_MODEL = "outputs/trained_model_full/best_model_long.pt"
OUTPUT_DIR = "outputs/model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 20

def load_model(model_path):
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True
    )
    model.decode_head.classifier = torch.nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return processor, model

print("加载新旧模型...")
old_processor, old_model = load_model(OLD_MODEL)
new_processor, new_model = load_model(NEW_MODEL)
print("✅ 两个模型都加载成功")

# 找图片
img_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:NUM_SAMPLES]
ious = {"old": [], "new": []}

for f in img_files:
    img_path = os.path.join(IMAGE_DIR, f)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 真实标注
    label_path = os.path.join(GT_DIR, f.replace('.jpg', '.png'))
    if not os.path.exists(label_path):
        continue
    gt = np.array(Image.open(label_path).resize((w, h), Image.NEAREST))
    gt_binary = (gt >= 1).astype(np.uint8)

    # 推理函数
    def predict(model, processor, img):
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        return cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    pred_old = predict(old_model, old_processor, img)
    pred_new = predict(new_model, new_processor, img)

    # 计算 IoU
    for name, pred in [("old", pred_old), ("new", pred_new)]:
        inter = np.logical_and(pred, gt_binary).sum()
        union = np.logical_or(pred, gt_binary).sum()
        iou = inter / union if union > 0 else 1.0
        ious[name].append(iou)

print("\n" + "="*60)
print("📊 新旧模型对比")
print("="*60)
print(f"旧模型 IoU: {np.mean(ious['old']):.4f} ± {np.std(ious['old']):.4f}")
print(f"新模型 IoU: {np.mean(ious['new']):.4f} ± {np.std(ious['new']):.4f}")
print(f"提升: {(np.mean(ious['new']) - np.mean(ious['old'])) * 100:.2f}%")
print("="*60)