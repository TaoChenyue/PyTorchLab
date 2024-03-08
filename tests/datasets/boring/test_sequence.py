def test_SequenceDataset():
    from torch import tensor

    from pytorchlab.datasets.boring.sequence import SequenceDataset

    sequences = range(10)
    dataset = SequenceDataset(sequences, transform=lambda x: tensor(x))
    assert len(dataset) == len(sequences)
    print("dataset[0].shape", dataset[0])
