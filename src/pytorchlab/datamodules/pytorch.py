from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from pytorchlab.datamodules.abstract.vision import (TorchVisionDataModule,
                                                    VisionDataModule)


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


class MNISTDataModule(TorchVisionDataModule):
    dataset_cls = datasets.MNIST


class FashionMNISTDataModule(TorchVisionDataModule):
    dataset_cls = datasets.FashionMNIST


class CIFAR10DataModule(TorchVisionDataModule):
    dataset_cls = datasets.CIFAR10


class CIFAR100DataModule(TorchVisionDataModule):
    dataset_cls = datasets.CIFAR100
