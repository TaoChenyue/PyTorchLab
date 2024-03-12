import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorchlab.callbacks.loss import LossCallback
from pytorchlab.datamodules.from_datasets import DataModule
from pytorchlab.datasets.split import SplitDataset
from pytorchlab.modules.lenet import LeNet5


def test_DataModule():
    main(epochs=1, limit_batches=1)


def main(
    root="dataset",
    epochs: int = 100,
    transform=transforms.ToTensor(),
    limit_batches: int | float | None = None,
):
    mnist_train_dataset = MNIST(
        root=root, train=True, download=True, transform=transform
    )
    mnist_test_dataset = MNIST(
        root=root, train=False, download=True, transform=transform
    )

    datamodule = DataModule(
        train_datasets=SplitDataset(mnist_train_dataset),
        val_datasets=SplitDataset(mnist_test_dataset, train=False),
        test_datasets=mnist_test_dataset,
        predict_datasets=mnist_test_dataset,
    )

    model = LeNet5()

    trainer = Trainer(
        max_epochs=epochs,
        devices=int(torch.cuda.is_available()),
        callbacks=[LossCallback()],
        logger=[TensorBoardLogger("lightning_logs/test_lenet", "mnist")],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_predict_batches=limit_batches,
        limit_test_batches=limit_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    main(epochs=10)
