import functools

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from pytorchlab.callbacks.image import ImageCallback
from pytorchlab.callbacks.loss import LossCallback
from pytorchlab.datamodules.from_datasets import DataModule
from pytorchlab.datasets.anomaly import AnomalyDataset
from pytorchlab.datasets.split import SplitDataset
from pytorchlab.models.decoder.convtranspose import SequentialConvTranspose2dBlock
from pytorchlab.models.encoder.conv import SequentialConv2dBlock
from pytorchlab.modules.ganomaly import GANomalyGenerator, GANomalyModule


def main(root="dataset", epochs: int = 100, limit_batches: int | float | None = None):
    model = GANomalyModule(
        generator=GANomalyGenerator(
            encoder=SequentialConv2dBlock(
                paras=[
                    (3, 64, 4, 2, 1),
                    (64, 128, 4, 2, 1),
                    (128, 64, 4, 2, 1),
                ]
            ),
            decoder=SequentialConvTranspose2dBlock(
                paras=[
                    (64, 128, 4, 2, 1),
                    (128, 64, 4, 2, 1),
                    (64, 3, 4, 2, 1),
                ]
            ),
            encoder2=SequentialConv2dBlock(
                paras=[
                    (3, 64, 4, 2, 1),
                    (64, 128, 4, 2, 1),
                    (128, 64, 4, 2, 1),
                ]
            ),
        ),
        discriminator=SequentialConv2dBlock(
            paras=[
                (6, 64, 4, 2, 1),
                (64, 64, 4, 2, 1),
                (64, 1, 4, 2, 1),
            ]
        ),
        optimizer_g=functools.partial(torch.optim.Adam, lr=0.0001, betas=(0.5, 0.999)),
        optimizer_d=functools.partial(torch.optim.Adam, lr=0.0001, betas=(0.5, 0.999)),
    )

    transform = transforms.Compose(
        [
            transforms.Resize(1024),
            transforms.ToTensor(),
        ]
    )

    dataset = AnomalyDataset(
        root=f"{root}/mvtec_anomaly_detection/bottle/train",
        transform=transform,
    )

    train_dataset = SplitDataset(dataset, 0.2)
    val_dataset = SplitDataset(dataset, 0.2, train=False)
    test_dataset = AnomalyDataset(
        root=f"{root}/mvtec_anomaly_detection/bottle/test",
        transform=transform,
    )
    predict_dataset = test_dataset

    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=val_dataset,
        test_datasets=test_dataset,
        predict_datasets=predict_dataset,
        batch_size=2,
        num_workers=20,
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=[1],
        logger=[TensorBoardLogger("lightning_logs/test_ganomaly", "mvtec_bottle")],
        callbacks=[
            LossCallback(),
            ImageCallback(
                batch_range=(0, 10),
                batch_indexes=[0, 1],
                image_slice=(0, 4),
                nrow=2,
                padding=2,
            ),
        ],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        limit_predict_batches=limit_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    # trainer.test(model, datamodule=datamodule)
    # trainer.predict(model, datamodule=datamodule)


def test_GANomalyModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main()
