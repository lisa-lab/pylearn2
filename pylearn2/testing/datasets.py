""" Simple datasets to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class ArangeDataset(DenseDesignMatrix):
    """
    A dataset where example i is just the number i. Makes it easy to track
    which sets of examples are visited.

    Parameters
    ----------
    num_examples : WRITEME
    To see the other parameters, look at the DenseDesignMatrix class
    documentation
    """
    def __init__(self, num_examples, *args, **kwargs):
        X = np.zeros((num_examples, 1))
        X[:, 0] = np.arange(num_examples)
        super(ArangeDataset, self).__init__(X, *args, **kwargs)


def random_dense_design_matrix(rng, num_examples, dim, num_classes):
    """
    Creates a random dense design matrix that has class labels.

    Parameters
    ----------
    rng : numpy.random.RandomState
        The random number generator used to generate the dataset.
    num_examples : int
        The number of examples to create.
    dim : int
        The number of features in each example.
    num_classes : int
        The number of classes to assign the examples to.
        0 indicates that no class labels will be generated.
    """
    X = rng.randn(num_examples, dim)

    if num_classes:
        Y = rng.randint(0, num_classes, (num_examples, 1))
        y_labels = num_classes
    else:
        Y = None
        y_labels = None

    return DenseDesignMatrix(X=X, y=Y, y_labels=y_labels)


def random_one_hot_dense_design_matrix(rng, num_examples, dim, num_classes):
    X = rng.randn(num_examples, dim)

    idx = rng.randint(0, num_classes, (num_examples, ))
    Y = np.zeros((num_examples, num_classes))
    for i in xrange(num_examples):
        Y[i, idx[i]] = 1

    return DenseDesignMatrix(X=X, y=Y)


def random_one_hot_topological_dense_design_matrix(rng,
                                                   num_examples,
                                                   shape,
                                                   channels,
                                                   axes,
                                                   num_classes):

    dims = {'b': num_examples,
            'c': channels}

    for i, dim in enumerate(shape):
        dims[i] = dim

    shape = [dims[axis] for axis in axes]

    X = rng.randn(*shape)

    idx = rng.randint(0, num_classes, (num_examples,))
    Y = np.zeros((num_examples, num_classes))
    for i in xrange(num_examples):
        Y[i, idx[i]] = 1

    return DenseDesignMatrix(topo_view=X, axes=axes, y=Y)


def random_dense_design_matrix_for_regression(rng, num_examples,
                                              dim, reg_min, reg_max):
    """
    Creates a random dense design matrix for regression.

    Parameters
    ----------
    rng : numpy.random.RandomState
        The random number generator used to generate the dataset.
    num_examples : int
        The number of examples to create.
    dim : int
        The number of features in each example.
    """
    X = rng.randn(num_examples, dim)
    Y = rng.randint(reg_min, reg_max, (num_examples, 1))

    return DenseDesignMatrix(X=X, y=Y)