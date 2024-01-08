import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def cal_loader_mean_std(loader: DataLoader):
    """
    calculate mean and std of dataloader

    Args:
        loader (DataLoader): _description_

    Returns:
        _type_: _description_
    """
    data_sum: torch.Tensor = 0
    data_squared_sum: torch.Tensor = 0
    for data, _ in loader:
        data_sum += torch.mean(data, dim=[0, 2, 3])
        data_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
    length = len(loader)
    mean = data_sum / length
    std = (data_squared_sum / length - mean**2) ** 0.5
    return mean.tolist(), std.tolist()


def cal_dataset_mean_std(
    name: str = "MNIST",
    root: str = "dataset",
    batch_size: int = 64,
) -> tuple[list[float], list[float]]:
    """
    calculate mean and std of train/test dataset

    Args:
        name (str): name of dataset
        root (str): root path of dataset
        batch_size (int): size of one batch. Defaults to 64.

    Returns:
        tuple[list[float],list[float]]: mean std of train and test dataset
    """
    dataset_cls = getattr(torchvision.datasets, name)
    assert issubclass(dataset_cls, Dataset), f"name:{name} not in torchvision.datasets"
    ans = []
    train_dataset = dataset_cls(
        root=root, train=True, download=False, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    ans.append(list(cal_loader_mean_std(train_loader)))
    test_dataset = dataset_cls(
        root=root, train=False, download=False, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ans.append(list(cal_loader_mean_std(test_loader)))
    return ans


if __name__ == "__main__":
    from jsonargparse import CLI

    print(CLI(cal_dataset_mean_std))
