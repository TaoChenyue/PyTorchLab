from pytorchlab.datasets.split import SplitDataset


def test_SplitDataset():

    a = list(range(10))
    d1 = SplitDataset(a, 0.5, seed=42, train=True)
    d2 = SplitDataset(a, 0.5, seed=42, train=False)
    assert len(d1) == 5 and len(d2) == 5
    print([x for x in d1], [x for x in d2])
