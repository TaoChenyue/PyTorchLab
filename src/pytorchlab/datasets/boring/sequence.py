from typing import Any, Callable, Sequence

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Any], transform: Callable | None = None):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index) -> Any:
        if self.transform is None:
            transform = lambda x: x
        else:
            transform = self.transform
        return transform(self.sequences[index])
