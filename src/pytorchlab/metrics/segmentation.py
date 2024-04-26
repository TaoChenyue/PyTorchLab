import torch
from torchmetrics import Metric

__all__ = [
    "SegmentationIOUMetric",
    "SegmentationDiceMetric",
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


class SegmentationDiceMetric(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("dice", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape, "a and b must have the same shape"
        if self.num_classes == 1:
            a_onehot = a
        else:
            a_onehot = torch.cat(
                [
                    torch.where(a == i, torch.ones_like(a), torch.zeros_like(a))
                    for i in range(self.num_classes)
                ],
                dim=-3,
            )
        b_onehot = torch.where(b < 0.5, torch.zeros_like(b), torch.ones_like(b))
        a_sum = torch.sum(a_onehot, dim=[-1, -2])
        b_sum = torch.sum(b_onehot, dim=[-1, -2])
        intersection_sum = torch.sum(a_onehot * b_onehot, dim=[-1, -2])
        self.dice += torch.mean((2 * intersection_sum + 1.0) / (a_sum + b_sum + 1.0))
        self.total += b.shape[0] if b.ndim == 4 else 1

    def compute(self):
        return self.dice / self.total
