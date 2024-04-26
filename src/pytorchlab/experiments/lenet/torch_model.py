import torch
from torch import nn

__all__ = [
    "LeNet5",
]


class LeNet5(nn.Module):
    def __init__(
        self,
        channel: int = 1,
        height: int = 28,
        width: int = 28,
        num_classes: int = 10,
    ):
        """
        LeNet5 model.

        Args:
            channel (int, optional): channel of input image. Defaults to 1.
            height (int, optional): height of input image. Defaults to 28.
            width (int, optional): width of input image. Defaults to 28.
            num_classes (int, optional): number of classes. Defaults to 10.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 6, 5, padding=2),  # (B,6,H,W)
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),  # (B,6,H/2,W/2)
            nn.Conv2d(6, 16, 5, padding=2),  # (B,16,H/2,W/2)
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),  # (B,16,H/4,W/4)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * (height // 4) * (width // 4), 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
