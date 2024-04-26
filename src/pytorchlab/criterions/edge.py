from abc import ABCMeta

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "_EdgeLoss",
    "SobelLoss",
    "LaplacianLoss",
]


class _EdgeLoss(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def get_edge(self, x):
        raise NotImplementedError

    def forward(self, x, y):
        edge_x = self.get_edge(x)
        edge_y = self.get_edge(y)
        return F.mse_loss(edge_x, edge_y, reduction="mean")


class SobelLoss(_EdgeLoss):
    def __init__(self):
        super().__init__()
        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_x = sobel_x.reshape((1, 1, 3, 3))
        sobel_y = sobel_y.reshape((1, 1, 3, 3))
        self.conv_hx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hx.weight = torch.nn.Parameter(sobel_x, requires_grad=False)
        self.conv_hy.weight = torch.nn.Parameter(sobel_y, requires_grad=False)

    def get_edge(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        hx = self.conv_hx(x)
        hy = self.conv_hy(x)
        return torch.abs(hx) + torch.abs(hy)


class LaplacianLoss(_EdgeLoss):
    def __init__(self):
        super().__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Laplacian = Laplacian.reshape((1, 1, 3, 3))
        self.conv_la = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_la.weight = torch.nn.Parameter(Laplacian, requires_grad=False)

    def get_edge(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        return self.conv_la(x)
