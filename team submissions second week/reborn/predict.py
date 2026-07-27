import torch
import numpy as np

from PIL import Image

from transformers import SegformerForSemanticSegmentation

import torchvision.transforms as transforms



DEVICE="cuda"



MODEL_PATH="./best_segformer_b0.pth"

IMAGE_PATH="./test.jpg"

OUTPUT_MASK="road_mask.png"

OUTPUT_OVERLAY="road_overlay.png"



model = SegformerForSemanticSegmentation.from_pretrained(
    "./models/segformer-b0",
    num_labels=2,
    ignore_mismatched_sizes=True
)



model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)



model.to(DEVICE)

model.eval()



transform = transforms.Compose([

    transforms.Resize(
        (288,512)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])



# 原图

origin = Image.open(
    IMAGE_PATH
).convert(
    "RGB"
)



w,h = origin.size



image = transform(
    origin
).unsqueeze(0).to(
    DEVICE
)



with torch.no_grad():


    output=model(
        pixel_values=image
    )


    logits=torch.nn.functional.interpolate(

        output.logits,

        size=(h,w),

        mode="bilinear",

        align_corners=False

    )


    pred=torch.argmax(
        logits,
        dim=1
    )[0]



mask=pred.cpu().numpy()



print(
    "mask values:",
    np.unique(mask)
)


print(
    "road pixels:",
    np.sum(mask==1)
)



# 保存黑白mask

mask_img=np.zeros(
    (h,w),
    dtype=np.uint8
)


mask_img[mask==1]=255



Image.fromarray(
    mask_img
).save(
    OUTPUT_MASK
)



# 透明叠加

origin_np=np.array(
    origin
).astype(
    np.float32
)



color=np.zeros_like(
    origin_np
)


color[:,:,1]=255



alpha=0.35



result=origin_np.copy()



road=mask==1



result[road]=(

    origin_np[road]*(1-alpha)

    +

    color[road]*alpha

)



result=np.clip(
    result,
    0,
    255
).astype(
    np.uint8
)



Image.fromarray(
    result
).save(
    OUTPUT_OVERLAY
)



print(
    "保存完成:"
)

print(
    OUTPUT_MASK
)

print(
    OUTPUT_OVERLAY
)