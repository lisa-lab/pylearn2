"""
This file tests the MNISTPlus class. majorly concerning the X and y member
of the dataset and their corresponding sizes, data scales and topological
views.
"""
from pylearn2.datasets.mnistplus import MNISTPlus
from pylearn2.space import IndexSpace, VectorSpace
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy as np


def test_MNISTPlus():
    """
    Test the MNISTPlus warper.
    Tests the scale of data, the splitting of train, valid, test sets.
    Tests that a topological batch has 4 dimensions.
    Tests that it work well with selected type of augmentation.
    """
    skip_if_no_data()
    for subset in ['train', 'valid', 'test']:
        ids = MNISTPlus(which_set=subset)
        assert 0.01 >= ids.X.min() >= 0.0
        assert 0.99 <= ids.X.max() <= 1.0
        topo = ids.get_batch_topo(1)
        assert topo.ndim == 4
        del ids

    train_y = MNISTPlus(which_set='train', label_type='label')
    assert 0.99 <= train_y.X.max() <= 1.0
    assert 0.0 <= train_y.X.min() <= 0.01
    assert train_y.y.max() == 9
    assert train_y.y.min() == 0
    assert train_y.y.shape == (train_y.X.shape[0], 1)

    train_y = MNISTPlus(which_set='train', label_type='azimuth')
    assert 0.99 <= train_y.X.max() <= 1.0
    assert 0.0 <= train_y.X.min() <= 0.01
    assert 0.0 <= train_y.y.max() <= 1.0
    assert 0.0 <= train_y.y.min() <= 1.0
    assert train_y.y.shape == (train_y.X.shape[0], 1)

    train_y = MNISTPlus(which_set='train', label_type='rotation')
    assert 0.99 <= train_y.X.max() <= 1.0
    assert 0.0 <= train_y.X.min() <= 0.01
    assert train_y.y.max() == 9
    assert train_y.y.min() == 0
    assert train_y.y.shape == (train_y.X.shape[0], 1)

    train_y = MNISTPlus(which_set='train', label_type='texture_id')
    assert 0.99 <= train_y.X.max() <= 1.0
    assert 0.0 <= train_y.X.min() <= 0.01
    assert train_y.y.max() == 9
    assert train_y.y.min() == 0
    assert train_y.y.shape == (train_y.X.shape[0], 1)
