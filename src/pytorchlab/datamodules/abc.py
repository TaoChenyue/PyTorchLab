from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, Literal

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pytorchlab.datamodules.utils import split_dataset


class ABCDataModule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        train_root: str | Path = "dataset",
        test_root: str | Path = "dataset",
        val_in_train: bool = True,
        val_split: int | float = 0.2,
        split_seed: int | None = 42,
        num_workers: int = 4,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: Callable | None = None,
        target_transforms: Callable | None = None,
        train_transforms: Callable | None = None,
        train_target_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        test_target_transforms: Callable | None = None,
    ) -> None:
        """
        Abstract datamodule class for every datamodule.

        Args:
            train_root (str | Path): Path of train dataset. Defaults to 'dataset'.
            test_root (str | Path): Path of test dataset. Defaults to 'dataset'.
            val_in_train (bool, optional): Validation dataset should be split from train dataset or not. Defaults to True.
            val_split (int | float, optional): Split length or rate of validation dataset. Defaults to 0.2.
            split_seed (int | None, optional): Seed for random split validation dataset. if None, it means split sequentially. Defaults to 42.
            num_workers (int, optional): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.
            batch_size (int, optional): Size of batch of one dataloader iteration. Defaults to 32.
            shuffle (bool, optional): Shuffle train dataset or not. Defaults to True.
            pin_memory (bool, optional): Put dataloader into device/CUDA. Defaults to True.
            drop_last (bool, optional): Drop the last incomplete batch or not. Defaults to False.
            transforms (Callable | None, optional): default transformations. Defaults to None.
            target_transforms (Callable | None, optional): default target transformations. Defaults to None.
            train_transforms (Callable | None, optional): train transformation. Defaults to None.
            train_target_transforms (Callable | None, optional): train target transformation. Defaults to None.
            test_transforms (Callable | None, optional): test transformation. Defaults to None.
            test_target_transforms (Callable | None, optional): test target transformation. Defaults to None.
        """
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
        """
        Entire train dataset

        Args:
            transforms (Callable): train transformation
            target_transforms (Callable): train target transformation

        Returns:
            Dataset: train dataset
        """

    @abstractmethod
    def entire_test_dataset(
        self,
        transforms: Callable,
        target_transforms: Callable,
    ) -> Dataset:
        """
        Entire test dataset

        Args:
            transforms (Callable): test transformation
            target_transforms (Callable): test target transformation

        Returns:
            Dataset: test dataset
        """

    @abstractmethod
    def default_transforms(self) -> Callable:
        """
        Default transformation set in class

        Returns:
            Callable: Default transformation
        """

    @abstractmethod
    def default_target_transforms(self) -> Callable:
        """
        Default target transformation set in class

        Returns:
            Callable: Default target transformation function
        """

    @property
    def transforms(self) -> Callable:
        """
        Default tranformation set by user. If None, use class default.

        Returns:
            Callable: Default transformation
        """
        return self._default_transforms or self.default_transforms()

    @property
    def target_transforms(self) -> Callable:
        """
        Default target transformation set by user. If None, use class default.

        Returns:
            Callable: Default target transformation function
        """
        return self._default_target_transforms or self.default_target_transforms()

    @property
    def train_transforms(self) -> Callable:
        """
        Train transformation. If None, use default transformation.

        Returns:
            Callable: Train transformation
        """
        return self._train_transforms or self.transforms

    @property
    def test_transforms(self) -> Callable:
        """
        Test transformation. If None, use default transformation.

        Returns:
            Callable: Test transformation
        """
        return self._test_transforms or self.transforms

    @property
    def train_target_transforms(self) -> Callable:
        """
        Train target transformation. If None, use default target transformation.

        Returns:
            Callable: Train target transformation
        """
        return self._train_target_transforms or self.target_transforms

    @property
    def test_target_transforms(self) -> Callable:
        """
        Test target transformation. If None, use default target transformation.

        Returns:
            Callable: Test target transformation
        """
        return self._test_target_transforms or self.target_transforms

    def setup(self, stage: Literal["fit", "validate", "test", "predict", None]):
        if self.val_in_train:
            if stage in ["fit", "validate", None]:
                dataset_train = self.entire_train_dataset(
                    transforms=self.train_transforms,
                    target_transforms=self.train_target_transforms,
                )
                self.dataset_train, self.dataset_val = split_dataset(
                    dataset=dataset_train,
                    split=self.val_split,
                    seed=self.split_seed,
                )
            if stage in ["test", None]:
                self.dataset_test = self.entire_test_dataset(
                    transforms=self.test_transforms,
                    target_transforms=self.test_target_transforms,
                )
        else:
            if stage in ["fit", None]:
                self.dataset_train = self.entire_train_dataset(
                    transforms=self.train_transforms,
                    target_transforms=self.train_target_transforms,
                )
            if stage in ["fit", "validate", "test", None]:
                dataset_test = self.entire_test_dataset(
                    transforms=self.test_transforms,
                    target_transforms=self.test_target_transforms,
                )
                self.dataset_val, self.dataset_test = split_dataset(
                    dataset=dataset_test,
                    split=1 - self.val_split,
                    seed=self.split_seed,
                )
        if stage in ["predict", None]:
            self.dataset_pred = self.entire_test_dataset(
                transforms=self.test_transforms,
                target_transforms=self.test_target_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Train dataloader

        Returns:
            DataLoader: Train dataloader
        """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """
        Validation dataloader

        Returns:
            DataLoader: Validation dataloader
        """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        """
        Test dataloader

        Returns:
            DataLoader: Test dataloader
        """
        return self._data_loader(self.dataset_test)

    def predict_dataloader(self) -> DataLoader:
        """
        Prediction dataloader

        Returns:
            DataLoader: Prediction dataloader
        """
        return self._data_loader(self.dataset_pred, batch_size=1)

    def _data_loader(
        self, dataset: Dataset, batch_size: int | None = None, shuffle: bool = False
    ):
        return DataLoader(
            dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
