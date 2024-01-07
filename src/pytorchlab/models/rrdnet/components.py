# see https://github.com/aaaaangel/RRDNet.git
import torch
import torch.nn as nn


class RRDNet(nn.Module):
    def __init__(self, in_channel: int):
        super(RRDNet, self).__init__()

        self.illumination_net = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

        self.reflectance_net = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, in_channel, 3, 1, 1),
        )

        self.noise_net = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, in_channel, 3, 1, 1),
        )

    def forward(self, input):
        illumination = torch.sigmoid(self.illumination_net(input))
        reflectance = torch.sigmoid(self.reflectance_net(input))
        noise = torch.tanh(self.noise_net(input))

        return illumination, reflectance, noise
