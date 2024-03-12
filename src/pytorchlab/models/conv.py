import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.type_hint import ModuleCallable


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
