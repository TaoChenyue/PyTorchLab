import torch
from torch import nn


class WeightedLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        return self.weight * self.criterion(*args, **kwargs)


def adaptive_weight_loss(criterions: list[torch.Tensor]) -> torch.Tensor:
    """
    Adaptive weight loss function.

    Args:
        criterions (list[torch.Tensor]): list of losses to be weighted.

    Returns:
        torch.Tensor: weighted loss.
    """
    total_loss = sum([crit.item() for crit in criterions])
    weights = [crit.item() / total_loss for crit in criterions]
    weighted_loss = sum([weight * loss for weight, loss in zip(weights, criterions)])
    return weighted_loss
