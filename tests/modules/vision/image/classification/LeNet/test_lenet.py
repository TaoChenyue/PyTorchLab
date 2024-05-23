import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from pytorchlab.callbacks import (
    ImageCallback,
    LabelCallback,
    LossCallback,
    MetricsClassificationCallback,
)
from pytorchlab.datamodules import DataModule
from pytorchlab.datasets import MNISTDataset, SplitDataset
from pytorchlab.modules import LeNet5Module


def main(
    root="dataset",
    epochs: int = 100,
    transform=transforms.ToTensor(),
    limit_batches: int | float | None = None,
):
    mnist_train_dataset = MNISTDataset(
        root=root, train=True, download=True, transform=transform
    )
    mnist_test_dataset = MNISTDataset(
        root=root, train=False, download=True, transform=transform
    )

    datamodule = DataModule(
        train_datasets=SplitDataset(mnist_train_dataset),
        val_datasets=SplitDataset(mnist_test_dataset, train=False),
        test_datasets=mnist_test_dataset,
        predict_datasets=mnist_test_dataset,
        batch_size=64,
        num_workers=20,
    )

    model = LeNet5Module()

    trainer = Trainer(
        max_epochs=epochs,
        devices=int(torch.cuda.is_available()),
        callbacks=[
            LossCallback(),
            ImageCallback(input_names=["image"], nrow=8),
            LabelCallback(
                input_names=["label"],
                output_names=["label"],
            ),
            MetricsClassificationCallback(name="label"),
        ],
        logger=[TensorBoardLogger("logs/lenet", "mnist")],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_predict_batches=limit_batches,
        limit_test_batches=limit_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


def test_LeNet5Module():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main(epochs=10)
