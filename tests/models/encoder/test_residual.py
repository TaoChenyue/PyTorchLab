import pytest
import torch

from pytorchlab.models.encoder.residual import ResidualBlock


@pytest.fixture
def t():
    return torch.randn(1, 3, 64, 64)


def test_ResidualBlock(t):
    block = ResidualBlock(
        3,
        64,
        stride=2,
    )
    out = block(t)
    assert out.shape == (1, 64, 32, 32)
