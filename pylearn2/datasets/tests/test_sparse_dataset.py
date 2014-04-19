"""
Unit tests for ../sparse_dataset.py
"""

import numpy as np
from pylearn2.datasets.sparse_dataset import SparseDataset
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.models.rbm import RBM
from scipy import *
from scipy.sparse import *

def test_iterator():
    """
    tests wether SparseDataset can be loaded and
    initializes iterator
    """

    data = array([1,2,3,4,5,6])
    row = array([0,0,1,2,2,2])
    col = array([0,2,2,0,1,2])
    x = csr_matrix( (data,(row,col)), shape=(3,3) ).todense()
    ds = SparseDataset(from_scipy_sparse_dataset=x)

    it = ds.iterator(mode='sequential', batch_size=1)
    it.next()

def test_training_a_model():
    """
    tests wether SparseDataset can be trained 
    with a dummy model.
    """

    data = array([1,2,3,4,5,6])
    row = array([0,0,1,2,2,2])
    col = array([0,2,2,0,1,2])
    x = csr_matrix( (data,(row,col)), shape=(3,3) ).todense()
    train = SparseDataset(from_scipy_sparse_dataset=x)

    rbm = RBM(nvis=3, nhid=3)
    trainer = DefaultTrainingAlgorithm(batch_size=1)
    try:
        trainer.setup(rbm, train)
    except:
        raise AssertionError("Could not train a dummy RBM model with sparce dataset")

if __name__ == '__main__':
    #test_iterator()
    test_training_a_model()
