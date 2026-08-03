import os
import sys
import torch
import numpy as np
from PIL import Image

from transformers import SegformerForSemanticSegmentation
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./best_segformer_b0.pth"

NUM_CLASSES = 2


def predict_image(model, image_path, save_dir):

    print("处理:", image_path)

    image = Image.open(image_path).convert("RGB")

    w, h = image.size


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


    x = transform(image).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        output = model(x).logits

        output = torch.nn.functional.interpolate(
            output,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        mask = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()


    road_pixels = np.sum(mask == 1)

    ratio = road_pixels / mask.size


    print("road pixels:", road_pixels)
    print("road ratio:", ratio)



    # 保存mask

    mask_img = (mask * 255).astype(np.uint8)

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]


    mask_path = os.path.join(
        save_dir,
        name + "_mask.png"
    )


    Image.fromarray(mask_img).save(mask_path)



    # overlay

    overlay = np.array(image).copy()

    overlay[mask == 1] = (
        overlay[mask == 1] * 0.5 +
        np.array([0,255,0]) * 0.5
    )


    overlay = Image.fromarray(
        overlay.astype(np.uint8)
    )


    overlay.save(
        os.path.join(
            save_dir,
            name+"_overlay.png"
        )
    )


    print("保存完成\n")





def main():

    if len(sys.argv)<2:

        print(
            "用法: python predict.py 图片文件夹"
        )

        return


    input_dir = sys.argv[1]


    save_dir="./results"

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    model = SegformerForSemanticSegmentation.from_pretrained(
        "./models/segformer-b0",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )


    checkpoint=torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint
    )


    model.to(DEVICE)

    model.eval()



    files=os.listdir(input_dir)


    images=[
        f for f in files
        if f.lower().endswith(
            (".jpg",".jpeg",".png")
        )
    ]


    print(
        "发现图片:",
        len(images)
    )


    for f in images:

        path=os.path.join(
            input_dir,
            f
        )

        predict_image(
            model,
            path,
            save_dir
        )



    print("全部完成")




if __name__=="__main__":
    main()