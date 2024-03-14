import functools

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorchlab.callbacks.images.gan import GANCallback
from pytorchlab.callbacks.loss import LossCallback
from pytorchlab.datamodules.from_datasets import DataModule
from pytorchlab.datasets.split import SplitDataset
from pytorchlab.models.encoder.linear import LinearBlock
from pytorchlab.modules.gan import GANModule


class Generator(nn.Module):
    def __init__(
        self, latent_dim: int, size: tuple[int, int, int] = (1, 28, 28)
    ) -> None:
        super().__init__()
        self.size = size
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 128),
            LinearBlock(128, functools.reduce(lambda x, y: x * y, size)),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = x.view(x.size(0), *self.size)
        return x


class Discriminator(nn.Module):
    def __init__(self, size: tuple[int, int, int] = (1, 28, 28)) -> None:
        super().__init__()
        self.model = nn.Sequential(
            LinearBlock(functools.reduce(lambda x, y: x * y, size), 128),
            LinearBlock(128, 1),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


def main(
    root="dataset",
    latent_dim: int = 100,
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

    model = GANModule(
        latent_dim=latent_dim,
        generator=Generator(latent_dim=latent_dim),
        discriminator=Discriminator(),
        optimizer_g=functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999)),
        optimizer_d=functools.partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999)),
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=int(torch.cuda.is_available()),
        callbacks=[LossCallback(), GANCallback(latent_dim=latent_dim)],
        logger=[TensorBoardLogger("lightning_logs/test_gan", "mnist")],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_predict_batches=limit_batches,
        limit_test_batches=limit_batches,
    )

    trainer.fit(model, datamodule=datamodule)


def test_GANModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main(epochs=10)
