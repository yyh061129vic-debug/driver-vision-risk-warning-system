import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # 类别权重
        # 0: 背景
        # 1: 道路

        self.register_buffer(
            "weight",
            torch.tensor(
                [0.3, 2.0],
                dtype=torch.float32
            )
        )


    def forward(self, logits, target):

        # 保证 weight 和 logits 类型一致
        weight = self.weight.to(
            device=logits.device,
            dtype=logits.dtype
        )


        ce = F.cross_entropy(
            logits,
            target,
            weight=weight
        )


        prob = torch.softmax(
            logits,
            dim=1
        )


        road_prob = prob[:,1]


        target = target.float()


        intersection = (
            road_prob * target
        ).sum()


        dice = (
            2 * intersection + 1
        ) / (
            road_prob.sum()
            +
            target.sum()
            +
            1
        )


        dice_loss = 1 - dice


        loss = (
            0.5 * ce
            +
            0.5 * dice_loss
        )


        return loss