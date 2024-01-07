from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from pytorchlab.datamodules.vision.abc import VisionDataModule


class MNISTDataModule(VisionDataModule):
    def prepare_data(self) -> None:
        datasets.MNIST(self.train_root, train=True, download=True)
        datasets.MNIST(self.test_root, train=False, download=True)

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.MNIST(
            self.train_root,
            train=True,
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.MNIST(
            self.test_root,
            train=False,
            transform=transforms,
            target_transform=target_transforms,
        )


class FashionMNISTDataModule(VisionDataModule):
    def prepare_data(self) -> None:
        datasets.FashionMNIST(self.train_root, train=True, download=True)
        datasets.FashionMNIST(self.test_root, train=False, download=True)

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.FashionMNIST(
            self.train_root,
            train=True,
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.FashionMNIST(
            self.test_root,
            train=False,
            transform=transforms,
            target_transform=target_transforms,
        )
