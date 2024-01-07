def test_ConvDiscriminator():
    import torch

    from pytorchlab.modules.gans.discriminator.conv import ConvDiscriminator

    g = ConvDiscriminator(
        channel=3,
    )

    x = torch.randn([32, 3, 64, 64])

    y = g(x)

    assert y.shape == torch.Size([32, 128, 4, 4])
