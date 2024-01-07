import pytest

data = [
    (100, 10, [90, 10]),
    (100, 0.1, [90, 10]),
    ("100", 10, TypeError),
    (100, "10", TypeError),
    (100, 10.1, ValueError),
    (100, -10, ValueError),
    (100, 109, ValueError),
]


@pytest.mark.parametrize("length, split, output", data)
def test_get_splits(length, split, output):
    from pytorchlab.datamodules.utils import get_splits

    if isinstance(output, list):
        assert get_splits(length, split) == tuple(output)
    else:
        with pytest.raises(Exception):
            get_splits(length, split)


dataset = [
    (list(range(10)), 0.5, None),
    (list(range(10)), 0.5, 42),
]


@pytest.mark.parametrize("dataset,split,seed", dataset)
def test_split_dataset(dataset, split, seed):
    from pytorchlab.datamodules.utils import split_dataset

    d1, d2 = split_dataset(dataset, split, seed)
    assert len(d1) == 5 and len(d2) == 5
    print("\n")
    print([x for x in d1], [x for x in d2])
