from torch import nn
from torch.nn import functional as F
from torchvision import models

__all__ = [
    "VGG16PerceptualLoss",
]


class VGG16PerceptualLoss(nn.Module):
    def __init__(self, nums_conv_layer: int = 4):
        super(VGG16PerceptualLoss, self).__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.modules = list(self.model.children())
        indexes = []
        for i, m in enumerate(self.modules):
            if "ReLU" in str(m):
                indexes.append(i)
        indexes.reverse()
        nums_conv_layer = min(len(indexes), nums_conv_layer)
        self.slices = []
        for i in range(nums_conv_layer):
            self.slices.append(nn.Sequential(*self.modules[: indexes[i]]))

    def forward(self, x, y):
        diff_list = []
        for i in range(len(self.slices)):
            x_f = self.slices[i](x)
            y_f = self.slices[i](y)
            diff_list.append(F.mse_loss(x_f, y_f))
        return sum(diff_list) / len(diff_list)
