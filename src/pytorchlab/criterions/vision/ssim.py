from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x, y):
        return 1.0 - self.ssim(x, y)
