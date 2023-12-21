from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets

from pytorchlab.datamodules.abstract.vision import VisionDataModule


class ImageFolderDataModule(VisionDataModule):
    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.ImageFolder(
            self.train_root.as_posix(),
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.ImageFolder(
            self.test_root.as_posix(),
            transform=transforms,
            target_transform=target_transforms,
        )


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


class CelebADataModule(VisionDataModule):
    def __init__(self, target_type: list[str] | str = "attr", **kwargs):
        super().__init__(**kwargs)
        self.target_type = target_type

    def prepare_data(self) -> None:
        datasets.CelebA(
            self.train_root, split="all", target_type=self.target_type, download=True
        )

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return ConcatDataset(
            [
                datasets.CelebA(
                    self.train_root,
                    split="train",
                    target_type=self.target_type,
                    transform=transforms,
                    target_transform=target_transforms,
                ),
                datasets.CelebA(
                    self.train_root,
                    split="valid",
                    target_type=self.target_type,
                    transform=transforms,
                    target_transform=target_transforms,
                ),
            ]
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.CelebA(
            self.test_root,
            split="test",
            target_type=self.target_type,
            transform=transforms,
            target_transform=target_transforms,
        )
