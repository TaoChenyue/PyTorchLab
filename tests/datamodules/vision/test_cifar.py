def test_CIFAR10DataModule():
    from pytorchlab.datamodules.vision.cifar import CIFAR10DataModule

    dm = CIFAR10DataModule(
        train_root="dataset",
        test_root="dataset",
    )
    dm.prepare_data()
    dm.setup(None)


def test_CIFAR100DataModule():
    from pytorchlab.datamodules.vision.cifar import CIFAR100DataModule

    dm = CIFAR100DataModule(
        train_root="dataset",
        test_root="dataset",
    )
    dm.prepare_data()
    dm.setup(None)
