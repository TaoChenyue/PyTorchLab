def test_AutoEncoder2d():
    main()


def main():
    import torch

    from pytorchlab.modules.autoencoder.components import AutoEncoder2d

    model = AutoEncoder2d(
        in_channel=1,
        out_channel=1,
        depth=4,
    )
    input_tensor = torch.randn(1, 1, 32, 32)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 1, 32, 32)


if __name__ == "__main__":
    main()
