"""module for testing datasets.avicenna"""
import unittest
import numpy as np
from pylearn2.datasets.avicenna import Avicenna
from pylearn2.testing.skip import skip_if_no_data


def test_avicenna():
    """test that train/valid/test sets load (when standardize=False/true)."""
    skip_if_no_data()
    data = Avicenna(which_set='train', standardize=False)
    assert data.X.shape == (150205, 120)

    data = Avicenna(which_set='valid', standardize=False)
    assert data.X.shape == (4096, 120)

    data = Avicenna(which_set='test', standardize=False)
    assert data.X.shape == (4096, 120)

    # test that train/valid/test sets load (when standardize=True).
    data_train = Avicenna(which_set='train', standardize=True)
    assert data_train.X.shape == (150205, 120)

    data_valid = Avicenna(which_set='valid', standardize=True)
    assert data_valid.X.shape == (4096, 120)

    data_test = Avicenna(which_set='test', standardize=True)
    assert data_test.X.shape == (4096, 120)

    dt = np.concatenate([data_train.X, data_valid.X, data_test.X], axis=0)
    # Force double precision to compute mean and std, otherwise the test
    # fails because of precision.
    assert np.allclose(dt.mean(dtype='float64'), 0)
    assert np.allclose(dt.std(dtype='float64'), 1.)
