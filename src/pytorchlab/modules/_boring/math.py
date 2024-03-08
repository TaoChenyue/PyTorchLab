from typing import Sequence

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer


class LinearModule(LightningModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.hparams.in_features)
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1)

    def _step(
        self,
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        x, y = batch[:2]
        pred = self(x)
        loss = F.mse_loss(pred, y.reshape(-1, self.hparams.out_features))
        return loss

    def training_step(self, batch: Sequence[Tensor], batch_idx: int) -> Tensor:
        loss = self._step(batch, batch_idx)
        return loss

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        loss = self._step(batch, batch_idx, dataloader_idx)
        return loss

    def test_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        loss = self._step(batch, batch_idx, dataloader_idx)
        return loss

    def predict_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        x = batch[0]
        return self(x)
