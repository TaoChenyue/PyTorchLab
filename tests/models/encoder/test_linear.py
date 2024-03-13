import pytest
import torch

from pytorchlab.models.encoder.linear import LinearBlock


@pytest.fixture
def t():
    return torch.randn(1, 3)


def test_LinearBlock(t):
    block = LinearBlock(3, 64)
    out = block(t)
    assert out.shape == (1, 64)
