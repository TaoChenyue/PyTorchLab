from torchvision.transforms import ToTensor

from pytorchlab.datasets.classification.mnist import FashionMNISTDataset, MNISTDataset


def main(loader=MNISTDataset):
    dataset = loader(transform=ToTensor())
    x = dataset[0]
    print(x)


def test_MNISTDataset():
    main(MNISTDataset)


def test_FashionMNISTDataset():
    main(FashionMNISTDataset)


if __name__ == "__main__":
    main()
