def test_CGAN():
    from lightning.pytorch import Trainer

    from pytorchlab.datamodules.vision.mnist import MNISTDataModule
    from pytorchlab.models.cgan import CGAN

    dm = MNISTDataModule()
    model = CGAN(
        latent_dim=100,
        num_classes=10,
    )

    trainer = Trainer(
        fast_dev_run=True,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
    )
    trainer.fit(model=model, datamodule=dm)
