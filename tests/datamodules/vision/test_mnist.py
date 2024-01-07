def test_MNISTDataModule():
    pass

    from pytorchlab.datamodules.vision.mnist import MNISTDataModule

    batch_size = 5
    dm = MNISTDataModule(
        train_root="dataset",
        test_root="dataset",
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup(None)


def test_FashionMNISTDataModule():
    from pytorchlab.datamodules.vision.mnist import FashionMNISTDataModule

    batch_size = 5
    dm = FashionMNISTDataModule(
        train_root="dataset",
        test_root="dataset",
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup(None)
