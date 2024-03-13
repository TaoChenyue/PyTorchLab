import pytest
import torch

from pytorchlab.models.decoder.convtranspose import ConvTranspose2dBlock


@pytest.fixture
def t():
    return torch.randn(1, 3, 32, 32)


def test_ConvTranspose2dBlock(t):
    block = ConvTranspose2dBlock(3, 64)
    out = block(t)
    assert out.shape == (1, 64, 64, 64)
