import functools

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from pytorchlab.datasets import ImagePairDataset,SplitDataset
from pytorchlab.datamodules import DataModule
from pytorchlab.models import UNet2d,SequentialConv2dBlock
from pytorchlab.modules import Pix2PixModule
from pytorchlab.callbacks import LossCallback, ImageCallback, MetricsImageQualityCallback


def main(epochs=10):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    _train_dataset = ImagePairDataset(
        root="dataset/facades/train",
        A_name="A",
        B_name="B",
        transform=transform,
    )
    _test_dataset = ImagePairDataset(
        root="dataset/facades/test",
        A_name="A",
        B_name="B",
        transform=transform,
    )
    train_dataset = SplitDataset(
        dataset=_train_dataset,
    )
    val_dataset = SplitDataset(
        dataset=_train_dataset,
        train=False,
    )
    test_dataset = _test_dataset
    predict_dataset = _test_dataset
    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=val_dataset,
        test_datasets=test_dataset,
        predict_datasets=predict_dataset,
        batch_size=8,
        num_workers=20,
    )
    generator = UNet2d(
        in_channel=1,
        out_channel=3,
        depth=4,
        norm=torch.nn.BatchNorm2d,
    )
    discriminator = SequentialConv2dBlock(
        paras=[
            (4, 64, 4, 2, 1),
            (64, 128, 4, 2, 1),
            (128, 64, 4, 2, 1),
            (64, 1, 4, 1, 1),
        ],
        norm=torch.nn.BatchNorm2d,
    )
    model = Pix2PixModule(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=functools.partial(torch.optim.Adam, lr=2e-4),
        optimizer_d=functools.partial(torch.optim.Adam, lr=2e-4),
    )

    trainer = Trainer(
        logger=TensorBoardLogger("logs/pix2pix", "facades"),
        callbacks=[
            LossCallback(),
            ImageCallback(
                input_names=["image", "reconstruct"], output_names=["reconstruct"]
            ),
            MetricsImageQualityCallback(name="reconstruct"),
        ],
        max_epochs=epochs,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    trainer.predict(model=model, datamodule=datamodule)
    
def test_Pix2PixModule():
    main(epochs=1)


if __name__ == "__main__":
    main()
