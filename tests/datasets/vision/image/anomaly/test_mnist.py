from pytorchlab.datasets import MNISTAnomalyDataset


def main(root="dataset"):
    dataset = MNISTAnomalyDataset(
        root=root,
        train=True,
        normal_classes=[0],
    )
    assert dataset[0]["label"] == 0
    dataset = MNISTAnomalyDataset(
        root=root,
        train=False,
        normal_classes=[0],
    )
    for i in range(10):
        print(dataset[i]["label"])


if __name__ == "__main__":
    main()
