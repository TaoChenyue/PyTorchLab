import torch
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.criterions import SemanticDiceLoss
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
        num_classes: int,
        bilinear: bool = False,
        lr: float = 1e-4,
    ):
        """
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        DOI:
            - arxiv: https://arxiv.org/abs/1505.04597
            - Springer: https://doi.org/10.1007/978-3-319-24574-4_28

        Args:
            in_channels (int): number of channel in the input image
            num_classes (int): number of class in the output mask
            bilinear (bool, optional): use bilinear when upsample. Defaults to False.
            lr (float, optional): learning rate. Defaults to 1e-4.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.model = UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            bilinear=bilinear,
        )
        self.lr = lr
        self.colormap = RandomColormap(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCELoss()
        self.criterion_dice = SemanticDiceLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch: ImageDatasetItem):
        image = batch["image"]
        segmentation = batch["mask"]
        pred: torch.Tensor = self(image)
        pred = pred.sigmoid()
        loss = self.criterion(pred, segmentation) + self.criterion_dice(
            pred, segmentation
        )

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
                "mask_colormap": self.colormap(pred),
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
