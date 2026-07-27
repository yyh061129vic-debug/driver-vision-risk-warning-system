import os
from PIL import Image

import numpy as np

import torch

from torch.utils.data import Dataset

import torchvision.transforms as transforms



class RoadDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        size=(288,512)
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.size = size


        self.images = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.endswith(".jpg")
                or f.endswith(".png")
            ]
        )


        self.image_transform = transforms.Compose(
            [

                transforms.Resize(
                    size
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

            ]
        )



        self.mask_transform = transforms.Resize(
            size,
            interpolation=Image.NEAREST
        )



    def __len__(self):

        return len(self.images)



    def __getitem__(self,idx):


        img_name = self.images[idx]


        # -----------------------
        # image
        # -----------------------

        img_path=os.path.join(
            self.image_dir,
            img_name
        )


        image=Image.open(
            img_path
        ).convert(
            "RGB"
        )


        image=self.image_transform(
            image
        )



        # -----------------------
        # mask
        # -----------------------

        base=os.path.splitext(
            img_name
        )[0]


        mask_name=base+"_drivable_id.png"


        mask_path=os.path.join(
            self.mask_dir,
            mask_name
        )


        if not os.path.exists(mask_path):

            raise FileNotFoundError(
                f"mask不存在: {mask_path}"
            )



        mask=Image.open(
            mask_path
        )


        mask=self.mask_transform(
            mask
        )


        mask=np.array(
            mask
        )



        # -----------------------
        # 类别处理
        # -----------------------

        # drivable_id:
        # 0 背景
        # 1 道路

        mask=(mask>0).astype(
            np.int64
        )


        mask=torch.tensor(
            mask,
            dtype=torch.long
        )



        return image,mask