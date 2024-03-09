from typing import Mapping, Sequence

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from pytorchlab.modules.autoencoder.components import AutoEncoder2d
from pytorchlab.type_hint import ModuleCallable


class AutoEncoder2dModule(LightningModule):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        nf: int = 64,
        depth: int = 8,
        hold_depth: int = 3,
        norm: ModuleCallable | None = None,
        activation: nn.Module | None = None,
        out_activation: nn.Module | None = None,
        criterion: nn.Module | None = None,
    ):
        super().__init__()
        activation = nn.ReLU(inplace=True) if activation is None else activation
        out_activation = nn.Tanh() if out_activation is None else out_activation
        criterion = nn.MSELoss() if criterion is None else criterion

        self.model = AutoEncoder2d(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nf=nf,
            depth=depth,
            hold_depth=hold_depth,
            norm=norm,
            activation=activation,
            out_activation=out_activation,
        )
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0):
        x, y = batch[0:2]
        pred = self(x)
        loss = self.criterion(pred, y)
        return {"loss": loss, "output": pred}

    def training_step(self, batch: Sequence[Tensor], batch_idx: int):
        return self._step(batch, batch_idx)

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def predict_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)
