def test_ConvGenerator():
    import torch

    from pytorchlab.modules.gans.generator.conv import ConvGenerator

    g = ConvGenerator(
        channel=3,
        latent_dim=100,
    )

    x = torch.randn([32, 100])

    y = g(x)

    assert y.shape == torch.Size([32, 3, 64, 64])
