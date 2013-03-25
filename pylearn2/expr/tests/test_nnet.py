"""
Useful expressions common to many neural network applications.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
from theano import tensor as T

from pylearn2.expr.nnet import softmax_ratio
from pylearn2.utils import sharedX

def test_softmax_ratio():
    # Tests that the numerically stabilized version of the softmax ratio
    # matches the naive implementation, for small input values

    n = 3
    m = 4

    rng = np.random.RandomState([2013, 3, 23])

    Z_numer = sharedX(rng.randn(m,n))
    Z_denom = sharedX(rng.randn(m,n))

    numer = T.nnet.softmax(Z_numer)
    denom = T.nnet.softmax(Z_denom)

    naive = numer / denom
    stable = softmax_ratio(numer, denom)

    naive = naive.eval()
    stable = stable.eval()

    assert np.allclose(naive, stable)

