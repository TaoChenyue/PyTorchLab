from typing import Any, Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import Tensor, nn


class LeNet5(LightningModule):
    def __init__(
        self,
        channel: int = 1,
        height: int = 28,
        width: int = 28,
        num_classes: int = 10,
        criterion: nn.Module = lazy_instance(nn.CrossEntropyLoss),
    ):
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

        self.criterion = criterion

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _get_output(self, batch: Sequence[Tensor]):
        x = batch[0]
        pred = self(x)
        return pred

    def _get_loss(self, batch: Sequence[Tensor], pred: Tensor):
        y = batch[1]
        loss = self.criterion(pred, y)
        return loss

    def _step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0):
        pred = self._get_output(batch)
        loss = self._get_loss(batch, pred)
        return {"loss": loss, "outputs": {"vectors": [pred]}}

    def training_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def test_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def predict_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)
