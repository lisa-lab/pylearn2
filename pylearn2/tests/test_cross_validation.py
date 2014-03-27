"""Tests for cross validation module."""
__author__ = 'Steven Kearnes'

from pylearn2.cross_validation import DatasetKFold
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.testing.skip import skip_if_no_sklearn
import unittest
import numpy as np

class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        skip_if_no_sklearn()
        X = np.random.random((1000, 15))
        y = np.random.randint(2, size=1000)
        dataset = DenseDesignMatrix(X=X, y=y)
        dataset.convert_to_one_hot()
        self.dataset = dataset

    def test_dataset_k_fold(self):
        cv = DatasetKFold(self.dataset)
        for datasets in cv:
            import IPython
            IPython.embed()
            sys.exit()
