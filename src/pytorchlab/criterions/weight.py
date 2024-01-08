import torch
from torch import nn


class WeightedLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        weight: float = 1.0,
        name: str = "weighted_loss",
        choice: list[int] | None = None,
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.criterion = criterion
        self.choice = choice

    def forward(self, tensor_list: list[torch.Tensor], **kwargs):
        if self.choice is None:
            tmp_list = tensor_list
        else:
            tmp_list = [tensor_list[i] for i in self.choice]
        return self.weight * self.criterion(tmp_list, **kwargs)
