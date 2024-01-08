import torch
from torch import nn
from torch.nn import functional as F


class GradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_xr = a[:, :, 1:, :]
        a_xl = a[:, :, :-1, :]
        a_yd = a[:, :, :, 1:]
        a_yu = a[:, :, :, :-1]
        a_x = torch.abs(a_xl - a_xr)
        a_y = torch.abs(a_yd - a_yu)

        b_xr = b[:, :, 1:, :]
        b_xl = b[:, :, :-1, :]
        b_yd = b[:, :, :, 1:]
        b_yu = b[:, :, :, :-1]
        b_x = torch.abs(b_xl - b_xr)
        b_y = torch.abs(b_yd - b_yu)
        return F.mse_loss(a_x, b_x) + F.mse_loss(a_y, b_y)


class SobelLoss(nn.Module):
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

    def forward(self, X, Y):
        X = torch.mean(X, dim=1, keepdim=True)
        X_hx = self.conv_hx(X)
        X_hy = self.conv_hy(X)
        G_X = torch.abs(X_hx) + torch.abs(X_hy)
        Y = torch.mean(Y, dim=1, keepdim=True)
        Y_hx = self.conv_hx(Y)
        Y_hy = self.conv_hy(Y)
        G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)
        return F.mse_loss(G_X, G_Y, reduction="mean")


class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Laplacian = Laplacian.reshape((1, 1, 3, 3))
        self.conv_la = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_la.weight = torch.nn.Parameter(Laplacian, requires_grad=False)

    def forward(self, X, Y):
        X = torch.mean(X, dim=1, keepdim=True)
        X_la = self.conv_la(X)
        Y = torch.mean(Y, dim=1, keepdim=True)
        Y_la = self.conv_la(Y)
        return F.mse_loss(X_la, Y_la, reduction="mean")


class TVLoss(nn.Module):
    def __init__(self) -> None:
        """
        Total Variant Loss
        """
        super().__init__()

    def forward(self, a: torch.Tensor):
        a_xr = a[:, :, 1:, :]
        a_xl = a[:, :, :-1, :]
        a_yd = a[:, :, :, 1:]
        a_yu = a[:, :, :, :-1]
        return F.mse_loss(a_xr, a_xl) + F.mse_loss(a_yd, a_yu)
