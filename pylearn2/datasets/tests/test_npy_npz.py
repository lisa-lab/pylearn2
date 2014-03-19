from pylearn2.datasets.npy_npz import NpyDataset, NpzDataset
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy as np


def test_npy_npz():
    skip_if_no_data()
    npy = NpyDataset(file='test.npy')
    npy._deferred_load()
    npz = NpzDataset(file='test.npz', key='arr_0')
    assert np.all(npy.X == npz.X)
