from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from pytorchlab.typehints import ImageDatasetItem, OutputDict, OutputsDict

__all__ = ["LeNet5Module"]


class LeNet5Module(LightningModule):
    def __init__(
        self,
        channel: int = 1,
        height: int = 28,
        width: int = 28,
        num_classes: int = 10,
        criterion: nn.Module = lazy_instance(nn.CrossEntropyLoss),
    ):
        """
        _summary_

        OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(
                images={"image": batch["image"]},
                labels={"label": batch["label"]},
            ),
            outputs=OutputDict(labels={"label": pred}),
        )

        Args:
            channel (int, optional): _description_. Defaults to 1.
            height (int, optional): _description_. Defaults to 28.
            width (int, optional): _description_. Defaults to 28.
            num_classes (int, optional): _description_. Defaults to 10.
            criterion (nn.Module, optional): _description_. Defaults to lazy_instance(nn.CrossEntropyLoss).
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

        self.criterion = criterion

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _get_output(self, batch: ImageDatasetItem):
        x = batch["image"]
        pred = self(x)
        return pred

    def _get_loss(self, batch: ImageDatasetItem, pred: Tensor):
        y = batch["label"]
        loss = self.criterion(pred, y)
        return loss

    def _step(self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0):
        pred = self._get_output(batch)
        loss = self._get_loss(batch, pred)
        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(
                images={"image": batch["image"]},
                labels={"label": batch["label"]},
            ),
            outputs=OutputDict(labels={"label": pred}),
        )

    def training_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def validation_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def test_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx)

    def predict_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)
