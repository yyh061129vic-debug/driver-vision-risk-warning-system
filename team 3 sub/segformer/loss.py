import torch
import torch.nn as nn
import torch.nn.functional as F



class RoadLoss(nn.Module):

    def __init__(self):

        super().__init__()


        # 类别权重
        #
        # background
        # road

        self.register_buffer(
            "weight",
            torch.tensor(
                [
                    0.3,
                    2.0
                ],
                dtype=torch.float32
            )
        )



    def forward(
            self,
            logits,
            target
    ):


        weight=self.weight.to(
            device=logits.device,
            dtype=logits.dtype
        )


        # =====================
        # Cross Entropy
        # =====================

        ce=F.cross_entropy(
            logits,
            target,
            weight=weight
        )



        # =====================
        # Dice
        # =====================

        prob=torch.softmax(
            logits,
            dim=1
        )


        road_prob=prob[:,1]


        target_float=target.float()



        intersection=(
            road_prob *
            target_float
        ).sum()



        dice=(

            2*intersection+1

        )/(

            road_prob.sum()
            +
            target_float.sum()
            +
            1

        )



        dice_loss=1-dice



        # =====================
        # final loss
        #
        # CE控制边界
        # Dice保持道路连续
        #
        # =====================


        loss=(

            0.6*ce

            +

            0.4*dice_loss

        )


        return loss