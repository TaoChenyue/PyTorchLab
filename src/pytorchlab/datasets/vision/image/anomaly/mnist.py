from typing import Callable

from torch.utils.data import Dataset, Subset
from torchvision import datasets

from pytorchlab.typehints import ImageDatasetItem

__all__ = ["MNISTAnomalyDataset"]


class MNISTAnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        download: bool = False,
        normal_classes: list[int] = [0],
    ) -> None:
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.normal_classes = normal_classes
        if train:
            self.sub_dataset = self.get_normal_subset(self.dataset)
        else:
            self.sub_dataset = self.dataset

    def get_normal_subset(self, dataset: datasets.MNIST):
        indexes = []
        for i in range(len(dataset)):
            image, label = dataset[i]
            if label in self.normal_classes:
                indexes.append(i)
        return Subset(dataset, indexes)

    def __len__(self):
        return len(self.sub_dataset)

    def __getitem__(self, index):
        image, label = self.sub_dataset[index]
        if label in self.normal_classes:
            label = 0
        else:
            label = 1
        return ImageDatasetItem(image=image, label=label)
