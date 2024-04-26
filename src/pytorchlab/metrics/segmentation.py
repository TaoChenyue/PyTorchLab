import torch
from torchmetrics import Metric

__all__ = [
    "SegmentationIOUMetric",
]


class SegmentationIOUMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, a: torch.Tensor, b: torch.Tensor):
        self.iou += torch.sum(torch.where((a > 0.5) & (b > 0.5), 1, 0)) / torch.sum(
            torch.where((a > 0.5) | (b > 0.5), 1, 0)
        )
        self.total += b.shape[0] if b.ndim == 4 else 1

    def compute(self):
        return self.iou / self.total
