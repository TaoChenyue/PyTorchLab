from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pytorchlab.datamodules.utils import split_dataset


class BasicDataModule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        train_root: str | Path,
        test_root: str | Path,
        val_in_train: bool = True,
        val_split: int | float = 0.2,
        split_seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: Callable = None,
        target_transforms: Callable = None,
        train_transforms: Callable = None,
        train_target_transforms: Callable = None,
        test_transforms: Callable = None,
        test_target_transforms: Callable = None,
    ) -> None:
        super().__init__()

        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.val_in_train = val_in_train
        self.val_split = val_split
        self.split_seed = split_seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._default_transforms = transforms
        self._default_target_transforms = target_transforms
        self._train_transforms = train_transforms
        self._train_target_transforms = train_target_transforms
        self._test_transforms = test_transforms
        self._test_target_transforms = test_target_transforms

    @abstractmethod
    def entire_train_dataset(
        self,
        transforms: Callable,
        target_transforms: Callable,
    ) -> Dataset:
        ...

    @abstractmethod
    def entire_test_dataset(
        self,
        transforms: Callable,
        target_transforms: Callable,
    ) -> Dataset:
        ...

    @abstractmethod
    def default_transforms(self):
        ...

    @abstractmethod
    def default_target_transforms(self):
        ...

    @property
    def transforms(self) -> Callable:
        return self._default_transforms or self.default_transforms()

    @property
    def target_transforms(self) -> Callable:
        return self._default_target_transforms or self.default_target_transforms()

    @property
    def train_transforms(self):
        return self._train_transforms or self.transforms

    @property
    def test_transforms(self):
        return self._test_transforms or self.transforms

    @property
    def train_target_transforms(self):
        return self._train_target_transforms or self.target_transforms

    @property
    def test_target_transforms(self):
        return self._test_target_transforms or self.target_transforms

    def setup(self, stage: str):
        dataset_train = self.entire_train_dataset(
            transforms=self.train_transforms,
            target_transforms=self.train_target_transforms,
        )
        dataset_test = self.entire_test_dataset(
            transforms=self.test_transforms,
            target_transforms=self.test_target_transforms,
        )

        if self.val_in_train:
            if stage in ["fit", "validate", None]:
                self.dataset_train, self.dataset_val = split_dataset(
                    dataset=dataset_train,
                    val_split=self.val_split,
                    seed=self.split_seed,
                )
            elif stage in ["test", None]:
                self.dataset_test = dataset_test
        else:
            if stage in ["fit", None]:
                self.dataset_train = dataset_train
            elif stage in ["fit", "validate", "test", None]:
                self.dataset_val, self.dataset_test = split_dataset(
                    dataset=dataset_test,
                    val_split=1 - self.val_split,
                    seed=self.split_seed,
                )
        if stage in ["predict", None]:
            self.dataset_pred = dataset_test

    def train_dataloader(self):
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        return self._data_loader(self.dataset_test)

    def predict_dataloader(self):
        return self._data_loader(self.dataset_pred)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
