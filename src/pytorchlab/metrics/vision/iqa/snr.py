import torch
from torchmetrics import Metric


class SNR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("snr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numbers", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, image):
        mean = torch.mean(image, dim=(2, 3))
        std = torch.std(image, dim=(2, 3))
        self.snr += torch.sum(mean / std)
        self.numbers += image.size(0)

    def compute(self):
        return self.snr / self.numbers
