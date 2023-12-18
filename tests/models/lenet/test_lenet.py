from lightning.pytorch import Trainer
from pytorchlab.datamodules.pytorch import MNISTDataModule
from pytorchlab.models.lenet import LeNet5
from lightning.pytorch.loggers import TensorBoardLogger
from pytorchlab.models.lenet.callbacks import LeNetCallback

def test_lenet():
    dm = MNISTDataModule(
        train_root="dataset",
        test_root="dataset",
    )
    model = LeNet5()

    logger = TensorBoardLogger(
        save_dir="test_logs",
        name="lenet5",
    )
    
    lenetCallback = LeNetCallback()

    trainer = Trainer(
        max_epochs=1,
        devices=[0],
        logger=logger,
        callbacks=[
            lenetCallback,
        ],
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
