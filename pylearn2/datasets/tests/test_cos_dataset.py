"""Test code for cos dataset."""
import numpy
from pylearn2.datasets.cos_dataset import CosDataset
from pylearn2.testing.skip import skip_if_no_data


def test_cos_dataset():
    """Tests if the dataset generator yields the desired value."""
    skip_if_no_data()
    dataset = CosDataset()

    sample_batch = dataset.get_batch_design(batch_size=10000)
    assert sample_batch.shape == (10000, 2)
    assert sample_batch[:, 0].min() >= dataset.min_x
    assert sample_batch[:, 0].max() <= dataset.max_x
