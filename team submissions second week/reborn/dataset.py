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
        size=(288,512),
        augment=True
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.size = size

        self.augment = augment


        self.images = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.endswith(".jpg")
                or f.endswith(".png")
            ]
        )


        # 图片基础处理
        self.image_transform = transforms.Compose(
            [

                transforms.Resize(
                    size
                ),

                transforms.ToTensor(),


                # 模拟车辆遮挡
                transforms.RandomErasing(
                    p=0.3,
                    scale=(0.02,0.15),
                    ratio=(0.3,3.3)
                ),


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


        # mask必须最近邻插值
        self.mask_transform = transforms.Resize(
            size,
            interpolation=Image.NEAREST
        )


        # 颜色增强
        self.color_aug = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        )



    def __len__(self):

        return len(self.images)



    def __getitem__(self,idx):


        img_name = self.images[idx]


        # =====================
        # image
        # =====================

        img_path=os.path.join(
            self.image_dir,
            img_name
        )


        image=Image.open(
            img_path
        ).convert(
            "RGB"
        )


        # =====================
        # mask
        # =====================

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


        # =====================
        # 同步增强
        # =====================

        if self.augment:


            # 左右翻转
            if np.random.rand() < 0.5:

                image=image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

                mask=mask.transpose(
                    Image.FLIP_LEFT_RIGHT
                )


            # 光照变化
            image=self.color_aug(
                image
            )



        # =====================
        # resize
        # =====================

        image=self.image_transform(
            image
        )


        mask=self.mask_transform(
            mask
        )


        mask=np.array(
            mask
        )


        # =====================
        # 类别处理
        # =====================

        # BDD drivable:
        # 0 background
        # >0 road


        mask=(mask>0).astype(
            np.int64
        )


        mask=torch.tensor(
            mask,
            dtype=torch.long
        )


        return image,mask