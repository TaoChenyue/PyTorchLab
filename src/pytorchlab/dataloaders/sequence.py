from typing import Sequence


# From [issue](https://github.com/Lightning-AI/pytorch-lightning/discussions/11024https://github.com/Lightning-AI/pytorch-lightning/discussions/11024)
class SequentialLoader:
    def __init__(self, *dataloaders: Sequence):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader
