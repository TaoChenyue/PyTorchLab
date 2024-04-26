import torch
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.criterions import SegmentationDiceLoss
from pytorchlab.experiments.unet.torch_model import UNet
from pytorchlab.transforms import RandomColormap
from pytorchlab.typehints import ImageDatasetItem, OutputsDict

__all__ = [
    "UNetModule",
]


class UNetModule(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            bilinear=bilinear,
        )
        self.lr = lr
        self.colormap = RandomColormap(num_classes=out_channels)
        self.criterion = (
            nn.CrossEntropyLoss() if out_channels > 1 else nn.BCEWithLogitsLoss()
        )
        self.criterion_dice = SegmentationDiceLoss(num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch: ImageDatasetItem):
        image = batch["image"]
        segmentation = batch["mask"]
        pred: torch.Tensor = self(image)
        loss = self.criterion(pred, segmentation) + self.criterion_dice(
            pred, segmentation
        )

        if self.out_channels == 1:
            mask_colormap = torch.where(
                pred > 0.5,
                torch.ones_like(pred, dtype=torch.float32),
                torch.zeros_like(pred, dtype=torch.float32),
            )
        else:
            mask_colormap = self.colormap(pred.argmax(dim=-3, keepdim=True))

        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs={
                "image": image,
                "mask": segmentation,
                "mask_colormap": self.colormap(segmentation),
            },
            outputs={
                "mask": pred,
                "mask_colormap": mask_colormap,
            },
        )

    def training_step(self, batch: ImageDatasetItem, batch_idx: int):
        return self._step(batch)

    def validation_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch)

    def test_step(
        self, batch: ImageDatasetItem, batch_idx: int, dataloader_idx: int = 0
    ):
        return self._step(batch)
