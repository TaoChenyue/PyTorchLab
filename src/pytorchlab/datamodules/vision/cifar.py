from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from pytorchlab.datamodules.vision.abc import VisionDataModule


class CIFAR10DataModule(VisionDataModule):
    def prepare_data(self) -> None:
        datasets.CIFAR10(self.train_root, train=True, download=True)
        datasets.CIFAR10(self.test_root, train=False, download=True)

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.CIFAR10(
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
        return datasets.CIFAR10(
            self.test_root,
            train=False,
            transform=transforms,
            target_transform=target_transforms,
        )


class CIFAR100DataModule(VisionDataModule):
    def prepare_data(self) -> None:
        datasets.CIFAR100(self.train_root, train=True, download=True)
        datasets.CIFAR100(self.test_root, train=False, download=True)

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.CIFAR100(
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
        return datasets.CIFAR100(
            self.test_root,
            train=False,
            transform=transforms,
            target_transform=target_transforms,
        )
