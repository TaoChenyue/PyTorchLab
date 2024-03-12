from torchvision import transforms

from pytorchlab.datasets.image_pair import ImagePairDataset


def main():
    dataset = ImagePairDataset(
        root="dataset/facades/train",
        A_name="A",
        B_name="B",
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    print(len(dataset), [x.shape for x in dataset[0]])


def test_ImagePairDataset():
    main()


if __name__ == "__main__":
    main()
