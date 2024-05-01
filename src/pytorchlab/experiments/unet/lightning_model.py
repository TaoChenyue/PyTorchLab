import torch
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.experiments.unet.torch_model import Unet
from pytorchlab.transforms import RandomColormap
from pytorchlab.typehints import ImageDatasetItem, OutputsDict

__all__ = [
    "UnetModule",
]


class UnetModule(LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: list[int] = [64, 128, 256, 512],
        bilinear: bool = False,
        lr: float = 1e-4,
    ):
        """
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        DOI:
            - arxiv: https://arxiv.org/abs/1505.04597
            - Springer: https://doi.org/10.1007/978-3-319-24574-4_28

        Args:
            in_channels (int, optional): number of channel in the input image. Defaults to 3.
            num_classes (int, optional): number of class in the output mask. Defaults to 1.
            features (list[int], optional): features of unet. Defaults to [64, 128, 256, 512].
            bilinear (bool, optional): use bilinear when upsample. Defaults to False.
            lr (float, optional): learning rate. Defaults to 1e-4.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.model = Unet(
            in_channels=in_channels,
            out_channels=num_classes,
            features=features,
            bilinear=bilinear,
        )
        self.lr = lr
        self.colormap = RandomColormap(num_classes=num_classes)
        self.criterion = (
            nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch: ImageDatasetItem):
        image = batch["image"]
        segmentation = batch["mask"]
        pred: torch.Tensor = self(image)
        loss = self.criterion(pred, segmentation)

        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs={
                "image": image,
                "mask": segmentation,
                "mask_colormap": self.colormap(segmentation),
            },
            outputs={
                "mask": pred.sigmoid(),
                "mask_colormap": self.colormap(pred.sigmoid()),
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
