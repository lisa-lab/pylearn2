from pylearn2.datasets.npy_npz import NpyDataset, NpzDataset
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy as np
import os


def test_npy_npz():
    skip_if_no_data()
    arr = np.array([[3, 4, 5], [4, 5, 6]])
    np.save('test.npy', arr)
    np.savez('test.npz', arr)
    npy = NpyDataset(file='test.npy')
    npy._deferred_load()
    npz = NpzDataset(file='test.npz', key='arr_0')
    assert np.all(npy.X == npz.X)
    os.remove('test.npy')
    os.remove('test.npz')
