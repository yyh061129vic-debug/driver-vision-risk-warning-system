import torch
import numpy as np
from PIL import Image
import random
import glob
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


MODEL_PATH = r"C:\Users\sirob\Desktop\code\internship\Models\segformer-b5-cityscapes"

processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


def segment_image(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    #Drivable Overlay
    road_mask = pred == 0

    img = np.array(image)

    overlay = img.copy()

    overlay[road_mask.cpu().numpy()] = [0, 255, 0]

    result = (
        img * 0.5 +
        overlay * 0.5
    ).astype(np.uint8)

    mask = road_mask.cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        result,
        contours,
        -1,
        (255,0,0),
        2
    )

    probs = F.softmax(outputs.logits, dim=1)
    road_confidence_map = F.interpolate(
    probs,
    size=image.size[::-1],   # (height, width)
    mode="bilinear",
    align_corners=False
    )[:, 0, :, :]

    road_confidence = road_confidence_map[0][road_mask].mean().item()

    return (
    img,
    result,
    mask,
    road_confidence_map[0].cpu(),
    road_confidence
    )



def process_video(input_video, output_video):

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    cv2.namedWindow("SegFormer Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SegFormer Video", 1280, 720)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Run SegFormer
        _, overlay, _, _, road_conf = segment_image(image)

        # Convert back to OpenCV format
        output = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Draw confidence
        cv2.putText(
            output,
            f"Road Confidence: {road_conf:.1%}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Save frame
        writer.write(output)

        # Show frame
        cv2.imshow("SegFormer Video", output)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()



def demo_images():

    image_paths = glob.glob(
        "Project_Files/data_raw/cityscapes/leftImg8bit/val/**/*.png",
        recursive=True
    )

    print(f"Found {len(image_paths)} images.")

    samples = random.sample(image_paths, 20)

    plt.figure(figsize=(18, 14))

    for i, path in enumerate(samples):

        image = Image.open(path).convert("RGB")

        _, overlay, _, _, road_conf = segment_image(image)

        plt.subplot(4, 5, i + 1)
        plt.imshow(overlay)
        plt.title(f"{road_conf:.2%}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def process_image(path):

    image = Image.open(path).convert("RGB")

    original, overlay, mask, conf_map, road_conf = segment_image(image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original
    axes[0,0].imshow(original)
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")

    # Overlay
    axes[0,1].imshow(overlay)
    axes[0,1].set_title(f"Drivable Area ({road_conf:.1%})")
    axes[0,1].axis("off")

    # Mask
    axes[1,0].imshow(mask, cmap="gray")
    axes[1,0].set_title("Pixel Mask")
    axes[1,0].axis("off")

    # Confidence
    heat = axes[1,1].imshow(
        conf_map,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1,1].set_title("Confidence")
    axes[1,1].axis("off")

    fig.colorbar(heat, ax=axes[1,1])

    plt.tight_layout()
    plt.show()



def main ():
    mode = "demo"
    # mode = "demo"
    # mode = "image"
    # mode = "video"

    if mode == "demo":

        demo_images()

    elif mode == "image":
        # enter image path here
        process_image("Project_Files/test_images/example.jpg")

    elif mode == "video":
        # enter video path here
        process_video(
            "Project_Files/data_raw/demo/testvid.mp4",
            "Project_Files/outputs/demo/demo.mp4"
    )
        
if __name__ == "__main__":
    main()

    #basically
    #training one takes literally 4 business days