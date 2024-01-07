def test_LinearDiscriminator():
    import torch

    from pytorchlab.modules.gans.discriminator.linear import \
        LinearDiscriminator

    g = LinearDiscriminator(
        channel=1,
        height=28,
        width=28,
    )
    x = torch.randn([32, 1, 28, 28])
    y = g(x)
    assert y.shape == torch.Size([32, 1])


def test_ConditionalLinearDiscriminator():
    import torch

    from pytorchlab.modules.gans.discriminator.linear import \
        ConditionalLinearDiscriminator

    g = ConditionalLinearDiscriminator(
        channel=1,
        height=28,
        width=28,
        num_classes=10,
    )

    img = torch.randn([32, 1, 28, 28])
    class_idx = torch.randint(0, 10, [32])

    y = g(img, class_idx)

    assert y.shape == torch.Size([32, 1])
