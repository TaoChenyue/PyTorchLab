import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.experiments.unet.torch_model import UNet
from pytorchlab.transforms import RandomColormap
from pytorchlab.typehints import ImageDatasetItem, ModuleCallable, OutputsDict

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
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            bilinear=bilinear,
        )
        self.criterion = (
            nn.CrossEntropyLoss() if out_channels > 1 else nn.BCEWithLogitsLoss()
        )
        self.lr = lr
        self.colormap = RandomColormap(num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch: ImageDatasetItem):
        image = batch["image"]
        segmentation = batch["mask"]
        pred:torch.Tensor = self(image)
        loss = self.criterion(pred, segmentation.squeeze(dim=-3))
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
                "mask_colormap": self.colormap(pred.argmax(dim=-3,keepdim=True)),
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import VOCSegmentation

    from pytorchlab.transforms import Image2SementicLabel, RandomColormap

    d = VOCSegmentation(
        root="dataset",
        image_set="train",
        download=False,
        transform=transforms.Compose(
            [
                transforms.CenterCrop((640, 640)),
                transforms.ToTensor(),
            ]
        ),
        target_transform=transforms.Compose(
            [
                transforms.CenterCrop((640, 640)),
                Image2SementicLabel(num_classes=21),
            ]
        ),
    )
    dd = DataLoader(
        d,
        batch_size=16,
    )
    model = UNetModule(3, 21)
    for x, y in dd:
        print(x.shape, y.shape)
        pred = model(x)
        print(pred.shape)
        print(nn.CrossEntropyLoss()(pred, y))
        break
