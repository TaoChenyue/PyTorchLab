def main(epochs: int = 100):
    import torch
    from lightning.pytorch import Trainer
    from torch import tensor

    from pytorchlab.callbacks.loss import LossCallback
    from pytorchlab.datamodules.from_datasets import DataModule
    from pytorchlab.datasets.boring.sequence import SequenceDataset
    from pytorchlab.modules._boring.math import LinearModule

    model = LinearModule(in_features=1, out_features=1)
    trainer = Trainer(max_epochs=epochs, devices=1, callbacks=[LossCallback()])

    fn = lambda x: 2.0 * x + 1.0

    transform = lambda x: [
        tensor(x, dtype=torch.float),
        tensor(fn(x), dtype=torch.float),
    ]

    train_dataset = SequenceDataset(
        range(0, 1000),
        transform=transform,
    )

    val_dataset0 = SequenceDataset(
        range(1000, 1100),
        transform=transform,
    )

    val_dataset1 = SequenceDataset(
        range(1100, 1200),
        transform=transform,
    )

    test_dataset = SequenceDataset(
        range(1200, 1300),
        transform=transform,
    )

    pred_dataset = SequenceDataset(
        range(1300, 1400),
        transform=transform,
    )

    datamodule = DataModule(
        train_datasets=train_dataset,
        val_datasets=[val_dataset0, val_dataset1],
        test_datasets=test_dataset,
        predict_datasets=pred_dataset,
        batch_size=20,
        num_workers=4,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)
    x = torch.tensor(1.0).to(model.device)
    pred = model(x)
    print(x, pred)


def test_DataModule():
    main(epochs=1)


if __name__ == "__main__":
    main(epochs=10)
