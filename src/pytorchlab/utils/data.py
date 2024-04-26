from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset, random_split

__all__ = [
    "split_dataset",
    "SplitDataset",
]


def split_dataset(
    dataset: Sequence,
    split: int | float,
    seed: int | None = None,
) -> tuple[Sequence, Sequence]:
    length = len(dataset)

    if isinstance(split, int):
        length_2 = split
        length_1 = length - split
    elif isinstance(split, float):
        length_2 = int(length * split)
        length_1 = length - length_2

    if seed is None:
        dataset_1 = Subset(dataset=dataset, indices=range(length_1))
        dataset_2 = Subset(dataset=dataset, indices=range(length_1, length))
    else:
        dataset_1, dataset_2 = random_split(
            dataset,
            [length_1, length_2],
            generator=torch.Generator().manual_seed(seed),
        )
    return dataset_1, dataset_2


class SplitDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        split: int | float = 0.2,
        seed: int | None = 42,
        train: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset1, self.dataset2 = split_dataset(
            dataset=dataset, split=split, seed=seed
        )
        self.train = train

    def __len__(self):
        return len(self.dataset1) if self.train else len(self.dataset2)

    def __getitem__(self, index: int):
        return self.dataset1[index] if self.train else self.dataset2[index]
