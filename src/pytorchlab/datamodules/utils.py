import torch
from torch.utils.data import Dataset, Subset, random_split


def get_splits(
    length: int,
    split: int | float,
) -> tuple[int, int]:
    """
    Split length into two parts depends on split

    Args:
        length (int): length
        split (int | float): split length or rate of the second part

    Returns:
        tuple[int,int]: length_1, length_2
    """
    if not isinstance(length, int):
        raise TypeError("length should be interger")
    elif length < 0:
        raise ValueError("length should be positive")

    if isinstance(split, int):
        if split < 0 or split > length:
            raise ValueError(
                f"split in int type should be between 0 and length:{length}"
            )
        length_2 = split
        length_1 = length - split
    elif isinstance(split, float):
        if split < 0 or split > 1:
            raise ValueError("split in float type should be between 0 and 1")
        length_2 = int(length * split)
        length_1 = length - length_2
    else:
        raise TypeError("split should be int or float")
    return length_1, length_2


def split_dataset(
    dataset: Dataset,
    split: int | float,
    seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into two parts depends on split

    Args:
        dataset (Dataset): dataset to be split
        split (int | float): split length or rate of the second dataset
        seed (int | None): if None, use sequential split, else use random split with seed. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: dataset_1, dataset_2
    """
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, split)
    if seed is None:
        dataset_1 = Subset(dataset=dataset, indices=range(splits[0]))
        dataset_2 = Subset(dataset=dataset, indices=range(splits[0], len_dataset))
    else:
        dataset_1, dataset_2 = random_split(
            dataset,
            splits,
            generator=torch.Generator().manual_seed(seed),
        )
    return dataset_1, dataset_2
