import pytest
import torch

from pytorchlab.models.encoder.conv import Conv2dBlock


@pytest.fixture
def t():
    return torch.randn(1, 3, 64, 64)


def test_Conv2dBlock(t):
    block = Conv2dBlock(3, 64, kernel_size=3, stride=1, padding=1)
    out = block(t)
    assert out.shape == (1, 64, 64, 64)
