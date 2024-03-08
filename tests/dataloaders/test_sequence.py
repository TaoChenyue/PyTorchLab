def test_SequentialLoader():
    from pytorchlab.dataloaders.sequence import SequentialLoader

    a = [1, 2, 3]
    b = [4, 5, 6]
    c = SequentialLoader(a, b)
    assert len(c) == len(a) + len(b)
