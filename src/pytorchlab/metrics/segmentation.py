import torch
from torchmetrics import Metric

__all__ = [
    "SemanticIOUMetric",
    "SemanticDiceMetric",
]


class SemanticIOUMetric(Metric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.threshold = threshold
        self.smooth = 1e-8

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_masks = torch.where(
            pred > self.threshold, torch.ones_like(pred), torch.zeros_like(pred)
        )
        target_masks = torch.where(
            target > self.threshold, torch.ones_like(target), torch.zeros_like(target)
        )
        intersection = torch.logical_and(pred_masks, target_masks).sum()
        union = torch.logical_or(pred_masks, target_masks).sum()
        self.intersection += intersection
        self.union += union

    def compute(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)


class SemanticDiceMetric(Metric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pred_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.threshold = threshold
        self.smooth = 1e-8

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_masks = torch.where(
            pred > self.threshold, torch.ones_like(pred), torch.zeros_like(pred)
        )
        target_masks = torch.where(
            target > self.threshold, torch.ones_like(target), torch.zeros_like(target)
        )
        intersection = torch.logical_and(pred_masks, target_masks).sum()
        self.intersection += intersection
        self.pred_sum += pred_masks.sum()
        self.target_sum += target_masks.sum()

    def compute(self):
        return (2 * self.intersection + self.smooth) / (
            self.pred_sum + self.target_sum + self.smooth
        )
