import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.typehints import ModuleCallable

__all__ = ["LinearBlock"]


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: ModuleCallable = nn.Identity,
        dropout: nn.Module = lazy_instance(nn.Identity),
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = norm(out_features)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x
