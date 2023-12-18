import torch
from torch.utils.data import Dataset, random_split


def get_splits(
    len_dataset: int,
    val_split: int | float,
) -> list[int]:
    """Computes split lengths for train and validation set."""
    assert isinstance(val_split, int | float), "val_split should be int or float type"
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        assert 0 <= val_split <= 1, "val_split in float type should between 0 and 1"
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    return splits


def split_dataset(
    dataset: Dataset,
    val_split: int | float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Splits dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    return random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
