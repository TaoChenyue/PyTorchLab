from typing import Iterable, Literal

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pytorchlab.utils.split_dataset import split_dataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset | Iterable[Dataset] | None = None,
        test_dataset: Dataset | Iterable[Dataset] | None = None,
        split: int | float = 0.2,
        seed: int | None = 42,
        num_workers: int = 0,
        batch_size: int = 1,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split = split
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: Literal["fit", "validate", "test", "predict", None]):
        if stage in ["fit", "validate", None]:
            if self.train_dataset is None:
                raise ValueError("train_dataset is not defined")
            elif isinstance(self.train_dataset, Dataset):
                self.dataset_train, self.dataset_val = split_dataset(
                    dataset=self.train_dataset,
                    split=self.split,
                    seed=self.seed,
                )
            else:
                self.dataset_train = []
                self.dataset_val = []
                for _ in self.train_dataset:
                    x, y = split_dataset(
                        dataset=_,
                        split=self.split,
                        seed=self.seed,
                    )
                    self.dataset_train.append(x)
                    self.dataset_val.append(y)

        if stage in ["test", None]:
            if self.test_dataset is None:
                raise ValueError("test_dataset is not defined")
            self.dataset_test = self.test_dataset

        if stage in ["predict", None]:
            if self.test_dataset is None:
                raise ValueError("test_dataset is not defined")
            self.dataset_predict = self.test_dataset

    def _dataloader(
        self,
        dataset: Dataset|Iterable[Dataset],
        shuffle: bool = False,
    ):
        if isinstance(dataset,Dataset):
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        return [DataLoader(
                x,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False,
            ) for x in dataset]

    def train_dataloader(self) -> DataLoader | Iterable[DataLoader]:
        return self._dataloader(self.dataset_train,shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader | Iterable[DataLoader]:
        return self._dataloader(self.dataset_val)

    def test_dataloader(self) -> DataLoader | Iterable[DataLoader]:
        return self._dataloader(self.dataset_test)

    def predict_dataloader(self) -> DataLoader | Iterable[DataLoader]:
        return self._dataloader(self.dataset_predict)
