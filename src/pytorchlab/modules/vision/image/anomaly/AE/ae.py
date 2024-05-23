import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.models import AutoEncoder2d
from pytorchlab.typehints import (
    ImageDatasetItem,
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
        lr: float = 1e-4,
    ):
        """
        AutoEncoder2dModule

        OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(
                images={"image": x},
                labels={"label": batch["label"]},
            ),
            outputs=OutputDict(
                images={"image": pred},
                metrics={
                    "score": anomaly_score,
                },
            ),
        )

        Args:
            in_channel (int): _description_
            out_channel (int): _description_
            kernel_size (int, optional): _description_. Defaults to 4.
            stride (int, optional): _description_. Defaults to 2.
            padding (int, optional): _description_. Defaults to 1.
            nf (int, optional): _description_. Defaults to 64.
            depth (int, optional): _description_. Defaults to 8.
            hold_depth (int, optional): _description_. Defaults to 3.
            norm (ModuleCallable, optional): _description_. Defaults to nn.Identity.
            down_activation (nn.Module, optional): _description_. Defaults to lazy_instance(nn.ReLU, inplace=True).
            up_activation (nn.Module, optional): _description_. Defaults to lazy_instance(nn.Tanh).
            criterion (nn.Module, optional): _description_. Defaults to lazy_instance(nn.MSELoss).
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "norm",
                "down_activation",
                "up_activation",
                "criterion",
            ]
        )
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
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def _step(self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        pred = self(x)
        loss = self.criterion(pred, x)
        heatmap = x - pred
        anomaly_score = torch.mean(
            torch.pow(heatmap, 2), dim=list(range(1, len(x.shape)))
        )
        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(
                images={"image": x},
                labels={"label": batch["label"]},
            ),
            outputs=OutputDict(
                images={"image": pred},
                metrics={
                    "score": anomaly_score,
                },
            ),
        )

    def training_step(self, batch: ImageDatasetItem, batch_idx: int):
        return self._step(batch, batch_idx)

    def validation_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def predict_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch, batch_idx, dataloader_idx=dataloader_idx)
