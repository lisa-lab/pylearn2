"""
Useful expressions common to many neural network applications.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as np
from theano import tensor as T

from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.expr.nnet import softmax_numpy
from pylearn2.expr.nnet import softmax_ratio
from pylearn2.expr.nnet import compute_recall
from pylearn2.utils import sharedX


def test_softmax_ratio():
    # Tests that the numerically stabilized version of the softmax ratio
    # matches the naive implementation, for small input values

    n = 3
    m = 4

    rng = np.random.RandomState([2013, 3, 23])

    Z_numer = sharedX(rng.randn(m, n))
    Z_denom = sharedX(rng.randn(m, n))

    numer = T.nnet.softmax(Z_numer)
    denom = T.nnet.softmax(Z_denom)

    naive = numer / denom
    stable = softmax_ratio(numer, denom)

    naive = naive.eval()
    stable = stable.eval()

    assert np.allclose(naive, stable)


def test_pseudoinverse_softmax_numpy():
    rng = np.random.RandomState([2013, 3, 28])

    p = np.abs(rng.randn(5))
    p /= p.sum()

    z = pseudoinverse_softmax_numpy(p)
    zbroad = z.reshape(1, z.size)
    p2 = softmax_numpy(zbroad)
    p2 = p2[0, :]

    assert np.allclose(p, p2)


def test_compute_recall():
    """
    Tests whether compute_recall function works as
    expected.
    """
    tp_pyval = 4
    ys_pyval = np.asarray([0, 1, 1, 0, 1, 1, 0])

    tp = sharedX(tp_pyval, name="tp")
    ys = sharedX(ys_pyval, name="ys_pyval")
    recall_py = tp_pyval / ys_pyval.sum()
    recall = compute_recall(ys, tp)
    assert np.allclose(recall.eval(),
                       recall_py)
