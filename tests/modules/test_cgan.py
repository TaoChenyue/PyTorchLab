import functools

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorchlab.callbacks.images.cgan import CGANCallback
from pytorchlab.callbacks.loss import LossCallback
from pytorchlab.datamodules.from_datasets import DataModule
from pytorchlab.datasets.split import SplitDataset
from pytorchlab.models.encoder.linear import LinearBlock
from pytorchlab.modules.cgan import CGANModule


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int = 10,
        size: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()
        self.size = size
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            LinearBlock(latent_dim + num_classes, 128),
            LinearBlock(128, functools.reduce(lambda x, y: x * y, size)),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        embedding = self.embedding(y)
        x = self.model(torch.cat([x, embedding], dim=-1))
        x = x.view(x.size(0), *self.size)
        return x


class Discriminator(nn.Module):
    def __init__(
        self, num_classes: int = 10, size: tuple[int, int, int] = (1, 28, 28)
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            LinearBlock(functools.reduce(lambda x, y: x * y, size) + num_classes, 128),
            LinearBlock(128, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        embedding = self.embedding(y)
        x = x.view(x.size(0), -1)
        x = self.model(torch.cat([x, embedding], dim=-1))
        return x


def main(
    root="dataset",
    latent_dim: int = 100,
    num_classes: int = 10,
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
        batch_size=64,
        num_workers=20,
    )

    model = CGANModule(
        latent_dim=latent_dim,
        num_classes=num_classes,
        generator=Generator(latent_dim=latent_dim, num_classes=num_classes),
        discriminator=Discriminator(num_classes=num_classes),
        optimizer_g=functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999)),
        optimizer_d=functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999)),
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=int(torch.cuda.is_available()),
        callbacks=[
            LossCallback(),
            CGANCallback(latent_dim=latent_dim, num_classes=num_classes),
        ],
        logger=[TensorBoardLogger("lightning_logs/test_cgan", "mnist")],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_predict_batches=limit_batches,
        limit_test_batches=limit_batches,
    )

    trainer.fit(model, datamodule=datamodule)


def test_CGANModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main(epochs=100)
