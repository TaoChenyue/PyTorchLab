import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.models import AutoEncoder2d
from pytorchlab.typehints import (
    ImageAnomalyItem,
    ModuleCallable,
    OutputDict,
    OutputsDict,
)


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
        norm: ModuleCallable = nn.Identity,
        down_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        up_activation: nn.Module = lazy_instance(nn.Tanh),
        criterion: nn.Module = lazy_instance(nn.MSELoss),
    ):
        super().__init__()
        self.model = AutoEncoder2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nf=nf,
            depth=depth,
            hold_depth=hold_depth,
            norm=norm,
            down_activation=down_activation,
            up_activation=up_activation,
        )
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _step(self, batch: ImageAnomalyItem, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        pred = self(x)
        loss = self.criterion(pred, x)
        anomaly_score = torch.mean(
            torch.pow(x - pred, 2), dim=list(range(1, len(x.shape)))
        )
        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(
                images={"image": x, "reconstructed": x},
                labels={"label": batch["label"]},
            ),
            outputs=OutputDict(images={"reconstructed": pred}),
            metrics={
                "anomaly_score": anomaly_score,
            },
        )

    def training_step(self, batch: ImageAnomalyItem, batch_idx: int):
        return self._step(batch, batch_idx)

    def validation_step(
        self, batch: ImageAnomalyItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_step(
        self, batch: ImageAnomalyItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def predict_step(
        self, batch: ImageAnomalyItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)
