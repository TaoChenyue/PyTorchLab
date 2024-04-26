from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.experiments.lenet.torch_model import LeNet5
from pytorchlab.typehints import ImageDatasetItem, OutputsDict

__all__ = ["LeNet5Module"]


class LeNet5Module(LightningModule):
    def __init__(
        self,
        channel: int = 1,
        height: int = 28,
        width: int = 28,
        num_classes: int = 10,
        criterion: nn.Module = lazy_instance(nn.CrossEntropyLoss),
        lr: float = 0.001,
    ):
        """
        Gradient-Based Learning Applied to Document Recognition
        DOI:
            - IEEE: https://doi.org/10.1109/5.726791

        Args:
            channel (int, optional): channel of input image. Defaults to 1.
            height (int, optional): height of input image. Defaults to 28.
            width (int, optional): width of input image. Defaults to 28.
            num_classes (int, optional): number of classes. Defaults to 10.
            criterion (nn.Module, optional): _description_. Defaults to lazy_instance(nn.CrossEntropyLoss).
            lr (float, optional): learning rate. Defaults to 0.001.
        """
        super().__init__()
        self.model = LeNet5(channel, height, width, num_classes)
        self.criterion = criterion
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        y = batch["label"]
        pred = self(x)
        loss = self.criterion(pred, y)
        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs={"image": batch["image"], "label": batch["label"]},
            outputs={"label": pred},
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
