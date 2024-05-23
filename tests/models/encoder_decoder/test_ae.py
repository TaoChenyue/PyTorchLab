import pytest
import torch

from pytorchlab.models.encoder_decoder import AutoEncoder2d, AutoEncoder2dBlock


@pytest.fixture
def t():
    return torch.randn(1, 3, 256, 256)


def test_AutoEncoder2dBlock(t):
    block = AutoEncoder2dBlock(
        last_channel=3,
        channel=64,
    )
    output = block(t)
    print(t.shape, output.shape)
    assert t.shape == output.shape


def test_AutoEncoder2d(t):
    model = AutoEncoder2d(
        in_channels=3,
        out_channels=3,
    )
    output = model(t)
    print(t.shape, output.shape)
    assert t.shape == output.shape
