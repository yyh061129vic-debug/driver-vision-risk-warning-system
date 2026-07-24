import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# 配置
# ============================================================
IMAGE_DIR = r"E:\PythonProject1\data\raw\bdd100k\100k\val"
MODEL_PATH = "outputs/trained_model/best_model.pt"
OUTPUT_DIR = "outputs/inference_refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_IMAGES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

# 加载模型（同上）
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True)
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", local_files_only=True)
model.decode_head.classifier = torch.nn.Conv2d(model.config.decoder_hidden_size, 2, kernel_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 边缘细化函数
def refine_mask(mask, kernel_size=5, iterations=1):
    """
    对二值掩码进行形态学闭运算（先膨胀后腐蚀），
    填补小孔、连接断裂区域，然后做开运算去除小噪点。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 闭运算：填洞
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    # 开运算：去噪
    refined = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return refined

# 推理并细化
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:NUM_IMAGES]
print(f"找到 {len(image_files)} 张图片")

for i, f in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, f)
    print(f"处理 [{i+1}/{NUM_IMAGES}]: {f}")

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 细化
    refined = refine_mask(pred, kernel_size=7, iterations=2)

    # 保存细化后的掩码（白色道路，黑色背景）
    save_path = os.path.join(OUTPUT_DIR, f"refined_{i+1:02d}.png")
    cv2.imwrite(save_path, refined * 255)
    print(f"   ✅ 已保存: {save_path}")

print(f"\n🎉 全部完成！细化结果保存在 {OUTPUT_DIR}")