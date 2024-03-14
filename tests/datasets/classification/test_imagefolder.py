from torchvision.transforms import ToTensor

from pytorchlab.datasets.classification.imagefolder import ImageFolderDataset


def main():
    dataset = ImageFolderDataset(
        root="dataset/facades/train",
        transform=ToTensor(),
    )
    x = dataset[0]
    print(x, dataset.dataset.imgs[0])
    assert x["label"] in range(10)


def test_ImageFolderDataset():
    main()


if __name__ == "__main__":
    main()
