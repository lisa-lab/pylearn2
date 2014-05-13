""" Simple datasets to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class ArangeDataset(DenseDesignMatrix):
    """
    A dataset where example i is just the number i. Makes it easy to track
    which sets of examples are visited.

    Parameters
    ----------
    num_examples : WRITEME
    """
    def __init__(self, num_examples):
        X = np.zeros((num_examples,1))
        X[:,0] = np.arange(num_examples)
        super(ArangeDataset, self).__init__(X)

def random_dense_design_matrix(rng, num_examples, dim, num_classes):
    X = rng.randn(num_examples, dim)

    if num_classes:
        Y = rng.randint(0, num_classes, (num_examples,1))
    else:
        Y = None

    return DenseDesignMatrix(X=X, y=Y)

def random_one_hot_dense_design_matrix(rng, num_examples, dim, num_classes):
    X = rng.randn(num_examples, dim)


    idx = rng.randint(0, num_classes, (num_examples,))
    Y = np.zeros((num_examples,num_classes))
    for i in xrange(num_examples):
        Y[i,idx[i]] = 1

    return DenseDesignMatrix(X=X, y=Y)

def random_one_hot_topological_dense_design_matrix(rng, num_examples, shape, channels, axes, num_classes):

    dims = {
            'b': num_examples,
            'c': channels
            }

    for i, dim in enumerate(shape):
        dims[i] = dim

    shape = [dims[axis] for axis in axes]

    X = rng.randn(*shape)

    idx = rng.randint(0, num_classes, (num_examples,))
    Y = np.zeros((num_examples,num_classes))
    for i in xrange(num_examples):
        Y[i,idx[i]] = 1

    return DenseDesignMatrix(topo_view=X, axes=axes, y=Y)
