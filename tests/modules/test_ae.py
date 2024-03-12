from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorchlab.callbacks.image import ImageCallback
from pytorchlab.callbacks.loss import LossCallback
from pytorchlab.datamodules.from_datasets import DataModule
from pytorchlab.datasets.split import SplitDataset
from pytorchlab.modules.ae import AutoEncoder2dModule
from pytorchlab.transforms.noise import GaussianNoise


class NoiseDataset(Dataset):
    def __init__(self, dataset: Dataset, mean: float = 0, std: float = 0.1) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = GaussianNoise(mean=mean, std=std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index][:2]
        noise_x = self.transform(x).detach()
        return noise_x, x, y


def main(root="dataset", epochs: int = 10, limit_batches: int | float | None = None):

    model = AutoEncoder2dModule(
        in_channel=1,
        out_channel=1,
        depth=4,
    )

    dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        ),
    )

    dataset = NoiseDataset(dataset, mean=0, std=0.1)

    train_dataset = SplitDataset(dataset, 0.2)
    val_dataset = SplitDataset(dataset, 0.2, train=False)
    test_dataset = MNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = NoiseDataset(test_dataset, mean=0, std=0.0)
    predict_dataset = test_dataset

    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=val_dataset,
        test_datasets=test_dataset,
        predict_datasets=predict_dataset,
        batch_size=128,
        num_workers=20,
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,
        logger=[TensorBoardLogger("lightning_logs/test_autoencoder2d", "mnist")],
        callbacks=[
            ImageCallback(
                batch_range=(0, 100),
                batch_indexes=[0, 1],
                image_slice=(0, 4),
                nrow=2,
                padding=2,
            ),
            LossCallback(),
        ],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        limit_predict_batches=limit_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


def test_AutoEncoder2dModule():
    main(epochs=1, limit_batches=1)


if __name__ == "__main__":
    main()
