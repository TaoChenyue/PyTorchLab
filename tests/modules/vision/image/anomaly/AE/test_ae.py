import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from pytorchlab.callbacks import ImageAnomalyCallback, ImageCallback, LossCallback
from pytorchlab.datamodules import DataModule
from pytorchlab.datasets import MNISTAnomalyDataset, SplitDataset
from pytorchlab.modules import AutoEncoder2dModule


def main(root: str = "dataset", epochs: int = 10, limit_batches: int | None = None):
    seed_everything(1234)
    model = AutoEncoder2dModule(
        in_channel=1,
        out_channel=1,
        depth=5,
    )
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = MNISTAnomalyDataset(
        root=root, train=True, download=True, transform=transform
    )
    _test_dataset = MNISTAnomalyDataset(
        root=root, train=False, download=True, transform=transform
    )
    val_dataset = _test_dataset
    test_dataset = _test_dataset
    predict_dataset = SplitDataset(_test_dataset, train=False)
    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=val_dataset,
        test_datasets=test_dataset,
        predict_datasets=predict_dataset,
        batch_size=64,
        num_workers=20,
    )
    trainer = Trainer(
        devices=int(torch.cuda.is_available()),
        max_epochs=epochs,
        logger=[TensorBoardLogger("logs/test_ae", "mnist_anomaly")],
        callbacks=[
            LossCallback(),
            ImageCallback(input_names=["image"], output_names=["image"]),
            ImageAnomalyCallback(),
        ],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_predict_batches=limit_batches,
        limit_test_batches=limit_batches,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


def test_AutoEncoder2dModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main(epochs=10)
