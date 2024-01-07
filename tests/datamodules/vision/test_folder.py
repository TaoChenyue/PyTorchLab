def test_ImageFolderDataModule():
    from pytorchlab.datamodules.vision.folder import ImageFolderDataModule

    dm = ImageFolderDataModule(
        train_root="dataset/NEU-CLS/train",
        test_root="dataset/NEU-CLS/test",
    )
    dm.prepare_data()
    dm.setup(None)
