import torch
from torch import nn

from pytorchlab.metrics import SegmentationDiceMetric

__all__ = [
    "SegmentationDiceLoss",
]


class SegmentationDiceLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.dice_metric = SegmentationDiceMetric(num_classes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return 1.0 - self.dice_metric(pred, target)
