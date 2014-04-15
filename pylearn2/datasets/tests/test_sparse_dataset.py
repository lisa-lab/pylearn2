"""
Unit tests for ../sparse_dataset.py
"""

import numpy as np
from pylearn2.datasets.sparse_dataset import SparseDataset


def test_iterator():
    """
    tests wether SparseDataset can be loaded and
    initializes iterator
    """

    x = np.ones((2, 3))
    ds = SparseDataset(from_scipy_sparse_dataset=x)

    it = ds.iterator(mode='sequential', batch_size=1)
    it.next()


if __name__ == '__main__':
    test_iterator()
