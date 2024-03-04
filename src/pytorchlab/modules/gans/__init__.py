from .linear import (
    LinearGenerator,
    LinearDiscriminator,
    ConditionalLinearGenerator,
    ConditionalLinearDiscriminator,
)
from .conv import ConvGenerator,ConvDiscriminator,NLayerDiscriminator,PixelDiscriminator
from .residual import ResidualGenerator
from .unet import UNetGenerator