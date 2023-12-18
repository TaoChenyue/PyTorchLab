import torch.nn as nn


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(
        self,
        channel: int,
        num_residual_blocks: int,
        base_features: int = 64,
    ):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        out_features = base_features
        model = [
            nn.ReflectionPad2d(channel),
            nn.Conv2d(channel, out_features, 2 * channel + 1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channel),
            nn.Conv2d(out_features, channel, 2 * channel + 1),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        base_features: int = 64,
    ):
        super(Discriminator, self).__init__()

        # self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channel, base_features, normalize=False),
            *discriminator_block(base_features, base_features * 2),
            *discriminator_block(base_features * 2, base_features * 4),
            *discriminator_block(base_features * 4, base_features * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_features * 8, 1, 4, padding=1),
        )

    def forward(self, img):
        return self.model(img)
