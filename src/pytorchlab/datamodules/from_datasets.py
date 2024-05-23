from typing import Callable, Iterable

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from pytorchlab.dataloaders import SequentialLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_datasets: Dataset | Iterable[Dataset] | None = None,
        val_datasets: Dataset | Iterable[Dataset] | None = None,
        test_datasets: Dataset | Iterable[Dataset] | None = None,
        predict_datasets: Dataset | Iterable[Dataset] | None = None,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Callable | None = None,
    ) -> None:
        """
        _summary_

        Args:
            train_datasets (Dataset | Iterable[Dataset] | None, optional): _description_. Defaults to None.
            val_datasets (Dataset | Iterable[Dataset] | None, optional): _description_. Defaults to None.
            test_datasets (Dataset | Iterable[Dataset] | None, optional): _description_. Defaults to None.
            predict_datasets (Dataset | Iterable[Dataset] | None, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 4.
            pin_memory (bool, optional): _description_. Defaults to True.
            drop_last (bool, optional): _description_. Defaults to False.
            collate_fn (Callable | None, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.train_datasets = self._datasets(train_datasets)
        self.val_datasets = self._datasets(val_datasets)
        self.test_datasets = self._datasets(test_datasets)
        self.predict_datasets = self._datasets(predict_datasets)
        self.collate_fn = collate_fn
        self.save_hyperparameters(
            ignore=[
                "train_datasets",
                "val_datasets",
                "test_datasets",
                "predict_datasets",
                "collate_fn",
            ]
        )

    def _datasets(
        self, datasets: Dataset | Iterable[Dataset] | None
    ) -> Iterable[Dataset]:
        if datasets is None:
            return []
        elif isinstance(datasets, Dataset):
            return [datasets]
        else:
            return [dataset for dataset in datasets]

    def _dataloader(
        self, datasets: Iterable[Dataset], shuffle: bool = False
    ) -> Iterable[DataLoader]:
        return [
            DataLoader(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=self.hparams.drop_last,
                persistent_workers=self.hparams.num_workers > 0,
                collate_fn=self.collate_fn,
            )
            for dataset in datasets
        ]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return SequentialLoader(*self._dataloader(self.train_datasets, shuffle=True))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader(self.val_datasets)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader(self.test_datasets)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader(self.predict_datasets)
