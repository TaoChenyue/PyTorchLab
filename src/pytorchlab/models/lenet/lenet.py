from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.type_hint import ModuleCallable

__all__ = ["LeNet5"]


class LeNet5(LightningModule):
    def __init__(
        self,
        channel: int = 1,
        height: int = 28,
        width: int = 28,
        num_classes: int = 10,
        criterion: ModuleCallable = nn.CrossEntropyLoss,
    ):
        super().__init__()
        # //////////////////////////////////////////////////
        # Model
        # (B,C,H,W)
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
        # //////////////////////////////////////////////////
        # Loss
        self.criterion = criterion()

    def forward(self, x):
        return self.fc(self.conv(x).view(x.shape[0], -1))

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # //////////////////////////////////////////////////
    # Train
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log_dict(
            {
                "loss": loss,
            },
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    # //////////////////////////////////////////////////
    # Validation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

    # //////////////////////////////////////////////////
    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

    # //////////////////////////////////////////////////
    # Predict
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        pred = self(x)
        return pred
