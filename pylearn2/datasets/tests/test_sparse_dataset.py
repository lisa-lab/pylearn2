"""
Unit tests for ../sparse_dataset.py
"""

import numpy as np
from pylearn2.datasets.sparse_dataset import SparseDataset
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.models.rbm import RBM
from scipy.sparse import csr_matrix


def test_iterator():
    """
    tests wether SparseDataset can be loaded and
    initializes iterator
    """

    x = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    ds = SparseDataset(from_scipy_sparse_dataset=x)
    it = ds.iterator(mode='sequential', batch_size=1)
    it.next()


def test_training_a_model():
    """
    tests wether SparseDataset can be trained
    with a dummy model.
    """

    x = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    train = SparseDataset(from_scipy_sparse_dataset=x)

    rbm = RBM(nvis=3, nhid=3)
    trainer = DefaultTrainingAlgorithm(batch_size=1)
    try:
        trainer.setup(rbm, train)
    except:
        message = "Could not train a dummy RBM model with sparce dataset"
        raise AssertionError(message)

if __name__ == '__main__':
    test_iterator()
    test_training_a_model()
