import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RoadDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        size=(384, 640),
        augment=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        self.images = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                )
            ]
        )

        self.image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_transform = transforms.Resize(
            size,
            interpolation=Image.NEAREST,
        )

        self.color_aug = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.2,
        )

    def __len__(self):
        return len(self.images)

    def random_shadow(self, image):
        if random.random() < 0.3:
            overlay = Image.new(
                "RGB",
                image.size,
                (0, 0, 0),
            )

            alpha = random.randint(25, 70)

            image = Image.blend(
                image,
                overlay,
                alpha / 255,
            )

        return image

    def random_scale(self, image, mask):
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)

            w, h = image.size
            nw = int(w * scale)
            nh = int(h * scale)

            image = image.resize(
                (nw, nh),
                Image.BILINEAR,
            )

            mask = mask.resize(
                (nw, nh),
                Image.NEAREST,
            )

            image = TF.center_crop(
                image,
                (h, w),
            )

            mask = TF.center_crop(
                mask,
                (h, w),
            )

        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(
            self.image_dir,
            img_name,
        )

        image = Image.open(
            img_path
        ).convert("RGB")

        base = os.path.splitext(img_name)[0]
        mask_name = base + "_drivable_id.png"

        mask_path = os.path.join(
            self.mask_dir,
            mask_name,
        )

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        mask = Image.open(mask_path)

        if self.augment:

            if random.random() < 0.5:
                image = image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

                mask = mask.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

            image, mask = self.random_scale(
                image,
                mask,
            )

            if random.random() < 0.25:
                params = transforms.RandomPerspective.get_params(
                    image.width,
                    image.height,
                    distortion_scale=0.15,
                )

                image = TF.perspective(
                    image,
                    *params,
                    interpolation=Image.BILINEAR,
                )

                mask = TF.perspective(
                    mask,
                    *params,
                    interpolation=Image.NEAREST,
                )

            image = self.color_aug(image)
            image = self.random_shadow(image)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = np.array(mask)

        # BDD drivable-area binary task:
        # 0 = background, >0 = road
        mask = (mask > 0).astype(np.int64)

        mask = torch.tensor(
            mask,
            dtype=torch.long,
        )

        return image, mask
