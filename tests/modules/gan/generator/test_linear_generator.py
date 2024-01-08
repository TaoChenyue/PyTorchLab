def test_LinearGenerator():
    import torch

    from pytorchlab.modules.gans.generator.linear import LinearGenerator

    g = LinearGenerator(
        channel=1,
        height=28,
        width=28,
        latent_dim=100,
    )
    x = torch.randn(32, 100)
    y = g(x)
    assert y.shape == torch.Size([32, 1, 28, 28])


def test_ConditionalLinearGenerator():
    import torch

    from pytorchlab.modules.gans.generator.linear import ConditionalLinearGenerator

    g = ConditionalLinearGenerator(
        channel=1,
        height=28,
        width=28,
        num_classes=10,
        latent_dim=100,
    )

    noise = torch.randn(32, 100)
    label = torch.randint(0, 10, (32,))
    y = g(noise, label)
    assert y.shape == torch.Size([32, 1, 28, 28])
