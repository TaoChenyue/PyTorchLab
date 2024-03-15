from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from pytorchlab.callbacks import ImageCallback, LossCallback, MetricsIQACallback
from pytorchlab.datamodules import DataModule
from pytorchlab.datasets import ImagePairDataset, SplitDataset
from pytorchlab.modules import AutoEncoder2dModule


def main(root="dataset", epochs: int = 10, limit_batches: int | float | None = None):

    model = AutoEncoder2dModule(
        in_channel=3,
        out_channel=3,
    )

    _train_dataset = ImagePairDataset(
        root=(Path(root) / "facades/train"),
        A_name="B",
        B_name="B",
        transform=transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        ),
    )
    _test_dataset = ImagePairDataset(
        root=(Path(root) / "facades/test"),
        A_name="B",
        B_name="B",
        transform=transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        ),
    )

    train_dataset = SplitDataset(_train_dataset, 0.2)
    val_dataset = SplitDataset(_train_dataset, 0.2, train=False)
    test_dataset = SplitDataset(_test_dataset, 0.2)
    predict_dataset = SplitDataset(_test_dataset, 0.2, train=False)

    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=val_dataset,
        test_datasets=test_dataset,
        predict_datasets=predict_dataset,
        batch_size=8,
        num_workers=20,
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=int(torch.cuda.is_available()),
        logger=[TensorBoardLogger("lightning_logs/test_autoencoder2d", "facades")],
        callbacks=[
            LossCallback(),
            ImageCallback(image_nums=4),
            MetricsIQACallback(),
        ],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        limit_predict_batches=limit_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


def test_AutoEncoder2dModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main(epochs=100)
