import cv2
import torch
import numpy as np
import os
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torchvision.transforms as transforms

def segment_image(img_bgr, model, transform, cmap, alpha=0.3):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)["out"]
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
    color_mask = cmap[pred]
    color_mask = cv2.resize(color_mask, (w, h))
    blend = cv2.addWeighted(img_bgr, 1-alpha, color_mask, alpha, 0)
    return blend

def main():
    device = torch.device("cpu")
    model = deeplabv3_mobilenet_v3_large(pretrained=True).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((520,520)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    # VOC21调色板，车辆红色，行人蓝色
    color_map = {
        0: [0,0,0],
        1: [128,0,0],2:[0,128,0],3:[128,128,0],4:[0,0,128],5:[128,0,128],
        6: [0,128,128],7:[255,0,0],8:[64,0,0],9:[192,0,0],10:[64,128,0],
        11:[192,128,0],12:[64,0,128],13:[192,0,128],14:[255,165,0],15:[0,0,255],
        16:[0,64,0],17:[128,64,0],18:[0,192,0],19:[128,192,0],20:[0,64,128]
    }
    cmap = np.array([color_map[i] for i in range(21)], dtype=np.uint8)

    video_path = "city_drive.mp4"  # 修改成你的视频文件名
    out_path = "./output_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
    cnt=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cnt += 1
        vis = segment_image(frame, model, transform, cmap)
        writer.write(vis)
        if cnt % 10 == 0:
            print(f"处理帧 {cnt}")
    cap.release()
    writer.release()
    print("✅ 完成")

if __name__ == "__main__":
    main()