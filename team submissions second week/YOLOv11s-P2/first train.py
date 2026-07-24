
BATCH_SIZE = 8
DEVICE = 0                       # GPU 号，没有 GPU 改成 "cpu"
WORKERS = 4
# =============================================================

CLASSES = [
    "bus", "traffic light", "traffic sign", "person", "bike",
        "truck", "motor", "car", "train", "rider",
        "animal", "cone", "lost_tire", "obstacle", "vehicle"
]

def convert_json_folder_to_yolo(json_dir, img_dir, out_dir):
    """
    智能转换：处理 frames.objects 结构，跳过纯 poly2d，跳过已有有效标签，
    但如果旧 .txt 为空则重新生成。
    """
    os.makedirs(out_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"在 {json_dir} 中发现 {len(json_files)} 个 JSON 文件")

    skipped = 0
    regenerated = 0
    converted = 0

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_name = data["name"]
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(out_dir, label_name)

        # 判断文件状态
        is_regenerated = False
        if os.path.exists(label_path) and os.path.getsize(label_path) > 10:
            skipped += 1
            continue
        elif os.path.exists(label_path):
            # 文件存在但为空（之前错误转换的），标记为重生成
            is_regenerated = True
            regenerated += 1

        # 读取图片真实尺寸
        img_path = os.path.join(img_dir, img_name)
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                w, h = im.size
        except:
            w, h = 1280, 720

        # 收集所有框对象
        objects = []
        if "frames" in data:
            for frame in data["frames"]:
                if "objects" in frame:
                    objects.extend(frame["objects"])
        elif "labels" in data:
            objects = data["labels"]

        # 写入 YOLO 格式
        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in objects:
                # 必须有 box2d 才处理（忽略只有 poly2d 的车道线等）
                if "box2d" not in obj:
                    continue
                cat = obj.get("category", "")
                if cat not in CLASSES:
                    continue
                cls_id = CLASSES.index(cat)
                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1
        status = "🔄 重生成" if is_regenerated else "✅ 新转换"
        print(f"  {status} {img_name} -> {label_path}")

    print(f"转换统计：新转换 {converted} 个，重生成空标签 {regenerated} 个，跳过有效标签 {skipped} 个\n")

def create_data_yaml():
    yaml_path = os.path.join(DATA_PATH, "data.yaml")
    content = f"""path: {DATA_PATH}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"已生成 {yaml_path}")

def create_model_yaml(nc):
    yaml_content = f"""# YOLO11s-P2 模型定义
nc: {nc}
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, Conv, [1024, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[19, 22, 25, 28], 1, Detect, [{nc}]]
"""
    with open("yolo11s-p2.yaml", 'w') as f:
        f.write(yaml_content.strip())
    print("已生成 yolo11s-p2.yaml\n")

def main():
    # 创建必要目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATA_PATH, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, "labels", split), exist_ok=True)

    # 智能转换 JSON -> TXT
    if CONVERT_JSON:
        print("=" * 50)
        print("处理训练集标签...")
        convert_json_folder_to_yolo(JSON_LABEL_DIR, JSON_IMG_DIR, OUTPUT_LABEL_DIR)

        val_json_dir = os.path.join(DATA_PATH, "labels", "val")
        val_img_dir = os.path.join(DATA_PATH, "images", "val")
        val_out_dir = os.path.join(DATA_PATH, "labels", "val")
        if os.path.exists(val_json_dir) and glob.glob(os.path.join(val_json_dir, "*.json")):
            print("处理验证集标签...")
            convert_json_folder_to_yolo(val_json_dir, val_img_dir, val_out_dir)
        else:
            print("未在 labels/val 中找到 .json，若已有有效 .txt 则忽略。")
        print("=" * 50 + "\n")

    create_data_yaml()
    create_model_yaml(len(CLASSES))

    from ultralytics import YOLO

    print("开始训练...")
    model = YOLO("yolo11s-p2.yaml")

    # 定义每轮覆盖保存模型的回调函数
    def save_epoch_model(trainer):
        epoch = trainer.epoch + 1
        save_dir = os.path.join(trainer.save_dir, "weights")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "latest_epoch.pt")
        ckpt = {
            'epoch': epoch,
            'model': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }
        torch.save(ckpt, save_path)
        print(f"✅ 已覆盖保存第 {epoch} 轮模型：{save_path}")

    # 注册回调
    model.add_callback("on_epoch_end", save_epoch_model)

    # 开始训练
    model.train(
        data=os.path.join(DATA_PATH, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        rect=True,
        close_mosaic=10,
        amp=True,
    )

    print("\n训练完成！")
    print("每轮覆盖保存的模型：runs/detect/train*/weights/latest_epoch.pt")
    print("验证集最优模型：runs/detect/train*/weights/best.pt")

if __name__ == "__main__":
    main()