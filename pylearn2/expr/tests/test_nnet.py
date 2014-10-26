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
import theano
from theano.gof.op import get_debug_values
from theano import tensor as T

from pylearn2.models.mlp import MLP, Sigmoid
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.expr.nnet import softmax_numpy
from pylearn2.expr.nnet import softmax_ratio
from pylearn2.expr.nnet import compute_recall
from pylearn2.expr.nnet import kl
from pylearn2.expr.nnet import elemwise_kl 
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


def test_kl():
    """
    Test whether function kl() has properly processed the input.
    """
    init_mode = theano.config.compute_test_value
    theano.config.compute_test_value = 'raise'
    
    try:
        mlp = MLP(layers=[Sigmoid(dim=10, layer_name='Y', irange=0.1)],
                  nvis=10)
        X = mlp.get_input_space().make_theano_batch()
        Y = mlp.get_output_space().make_theano_batch()
        X.tag.test_value = np.random.random(
            get_debug_values(X)[0].shape).astype(theano.config.floatX)
        Y_hat = mlp.fprop(X)

        # This call should not raise any error:
        ave = kl(Y, Y_hat, 1)

        # The following calls should raise ValueError exceptions:
        Y.tag.test_value[2][3] = 1.1
        np.testing.assert_raises(ValueError, kl, Y, Y_hat, 1)
        Y.tag.test_value[2][3] = -0.1
        np.testing.assert_raises(ValueError, kl, Y, Y_hat, 1)
    
    finally:
        theano.config.compute_test_value = init_mode


def test_elemwise_kl():
    """
    Test whether elemwise_kl() function has properly processed the
    input.
    """
    init_mode = theano.config.compute_test_value
    theano.config.compute_test_value = 'raise' 
    
    try:
        mlp = MLP(layers=[Sigmoid(dim=10, layer_name='Y', irange=0.1)], 
                  nvis=10)
        X = mlp.get_input_space().make_theano_batch()
        Y = mlp.get_output_space().make_theano_batch()
        X.tag.test_value = np.random.random(
            get_debug_values(X)[0].shape).astype(theano.config.floatX)
        Y_hat = mlp.fprop(X)

        # This call should not raise any error:
        ave = elemwise_kl(Y, Y_hat)

        # The following calls should raise ValueError exceptions:
        Y.tag.test_value[2][3] = 1.1
        np.testing.assert_raises(ValueError, elemwise_kl, Y, Y_hat)
        Y.tag.test_value[2][3] = -0.1
        np.testing.assert_raises(ValueError, elemwise_kl, Y, Y_hat)
    
    finally:
        theano.config.compute_test_value = init_mode

   
