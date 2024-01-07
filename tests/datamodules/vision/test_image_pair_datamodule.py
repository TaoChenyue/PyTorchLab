def test_ImagePairDatamodules():
    from torchvision import transforms

    from pytorchlab.datamodules.vision.image_pair import ImagePairDataModule

    tfs = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    dm = ImagePairDataModule(
        train_root="dataset/facades/train",
        test_root="dataset/facades/test",
        A_name="A",
        B_name="B",
        transforms=tfs,
    )

    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in dl:
        a, b = batch
        assert a.size(1) == 3
        assert b.size(1) == 3
        break
