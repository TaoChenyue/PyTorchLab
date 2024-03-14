from torchvision import transforms

from pytorchlab.datasets.anomaly import AnomalyDataset, MvtecMask


def main():
    dataset = AnomalyDataset(
        root="dataset/mvtec_anomaly_detection/bottle/test",
        transform=transforms.ToTensor(),
        get_mask=MvtecMask(transform=transforms.ToTensor()),
    )
    print(dataset[0], dataset[0]["label"], dataset[0]["mask"].max())


def test_AnomalyDataset():
    main()


if __name__ == "__main__":
    main()
