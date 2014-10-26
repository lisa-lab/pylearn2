import os
import pylearn2
from pylearn2.datasets.csv_dataset import CSVDataset
import numpy as np


def test_loading_classification():
    test_path = os.path.join(pylearn2.__path__[0],
                             'datasets', 'tests', 'test.csv')
    d = CSVDataset(path=test_path, expect_headers=False)
    assert(np.array_equal(d.X, np.array([[1., 2., 3.], [4., 5., 6.]])))
    assert(np.array_equal(d.y, np.array([0., 1.])))


def test_loading_regression():
    test_path = os.path.join(pylearn2.__path__[0],
                             'datasets', 'tests', 'test.csv')
    d = CSVDataset(path=test_path, task="regression", expect_headers=False)
    assert(np.array_equal(d.X, np.array([[1., 2., 3.], [4., 5., 6.]])))
    assert(np.array_equal(d.y, np.array([[0.], [1.]])))
