import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ============================================================
# 配置
# ============================================================
DATA_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\train"
OUTPUT_DIR = "outputs/lane_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# Cityscapes 19类颜色映射
COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

ROAD_CLASS_ID = 0

# ============================================================
# 查找图片
# ============================================================
def find_images(data_dir, num_images=20):
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.jpg'):
                all_files.append(os.path.join(root, f))
    return sorted(all_files)[:num_images]

# ============================================================
# 加载模型
# ============================================================
print("正在加载 SegFormer-B0 模型...")
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
).to(DEVICE)
model.eval()
print("✅ 模型加载成功！")

# ============================================================
# 处理单张图片
# ============================================================
def process_image(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img_pil)
    h, w = img_rgb.shape[:2]

    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    road_mask = (pred == ROAD_CLASS_ID).astype(np.uint8) * 255

    # ---- 平滑车道边界 ----
    boundary_img = img_rgb.copy()
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(boundary_img, [approx], -1, (0, 255, 0), 3)

    # ---- 彩色分割掩码 ----
    mask_color = np.zeros_like(img_rgb)
    for class_id, color in enumerate(COLORS):
        mask_color[pred == class_id] = color

    # ---- 道路区域高亮（绿色半透明） ----
    road_overlay = img_rgb.copy()
    road_pixels = road_mask > 0
    if np.sum(road_pixels) > 0:
        road_overlay[road_pixels] = (road_overlay[road_pixels] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)

    return {
        "original": img_rgb,
        "mask_color": mask_color,
        "road_overlay": road_overlay,
        "boundary": boundary_img,
        "road_mask": road_mask
    }

# ============================================================
# 可视化
# ============================================================
def visualize(result, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(result["original"])
    axes[0].set_title("1. 输入图像")
    axes[0].axis('off')

    axes[1].imshow(result["mask_color"])
    axes[1].set_title("2. 分割掩码 (19类)")
    axes[1].axis('off')

    axes[2].imshow(result["road_overlay"])
    axes[2].set_title("3. 道路区域 (绿色高亮)")
    axes[2].axis('off')

    axes[3].imshow(result["boundary"])
    axes[3].set_title("4. 平滑车道边界 (绿色线条)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    img_files = find_images(DATA_DIR, NUM_IMAGES)
    print(f"✅ 找到 {len(img_files)} 张图片，开始处理...")

    for idx, img_path in enumerate(img_files):
        print(f"处理 [{idx+1}/{len(img_files)}]: {os.path.basename(img_path)}")
        result = process_image(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"lane_{idx+1:02d}.png")
        visualize(result, save_path)

    print(f"\n✅ 全部完成！结果保存在: {OUTPUT_DIR}/")
    print("📊 四栏对比图说明：")
    print("  1. 输入图像")
    print("  2. 19类分割掩码")
    print("  3. 道路区域（绿色高亮）")
    print("  4. 平滑车道边界（绿色线条）")