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
