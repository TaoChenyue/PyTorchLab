def test_GAN():
    from lightning.pytorch import Trainer, seed_everything

    from pytorchlab.datamodules.vision.mnist import MNISTDataModule
    from pytorchlab.models.gan import GAN

    m = GAN()
    dm = MNISTDataModule()
    seed_everything(1234)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=m, datamodule=dm)
