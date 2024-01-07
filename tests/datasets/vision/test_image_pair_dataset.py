def test_ImagePairDataset():
    import torch
    from torchvision import transforms

    from pytorchlab.datasets.vision.image_pair import ImagePairDataset

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImagePairDataset(
        root="dataset/facades/train",
        A_name="A",
        B_name="B",
        transform=transform,
        mode_A="RGB",
        mode_B="RGB",
    )

    # 测试dataset的样本
    image_A, image_B = dataset[0]
    assert isinstance(image_A, torch.Tensor)
    assert isinstance(image_B, torch.Tensor)
    assert len(dataset) == 606 - 106
