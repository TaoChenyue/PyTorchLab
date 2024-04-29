import torch
from torch import nn

__all__ = [
    "SemanticDiceLoss",
]


class SemanticDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-8

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = torch.sum(pred_flat * target_flat)
        union = torch.sum(pred_flat) + torch.sum(target_flat)

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coefficient

        return dice_loss
