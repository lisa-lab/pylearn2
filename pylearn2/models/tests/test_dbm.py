__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np
assert hasattr(np, 'exp')

from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.expr.basic import is_binary
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.models.dbm import BinaryVector
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.models.dbm import DBM
from pylearn2.utils import sharedX

def check_binary_samples(value, expected_shape, expected_mean, tol):
    """
    Tests that a matrix of binary samples (observations in rows, variables
        in columns)
    1) Has the right shape
    2) Is binary
    3) Converges to the right mean
    """
    assert value.shape == expected_shape
    assert is_binary(value)
    mean = value.mean(axis=0)
    max_error = np.abs(mean-expected_mean).max()
    if max_error > tol:
        print 'Actual mean:'
        print mean
        print 'Expected mean:'
        print expected_mean
        print 'Maximal error:', max_error
        raise ValueError("Samples don't seem to have the right mean.")

def test_binary_vis_layer_make_state():

    # Verifies that BinaryVector.make_state creates
    # a shared variable whose value passes check_binary_samples

    n = 5
    num_samples = 1000
    tol = .04

    layer = BinaryVector(nvis = n)

    rng = np.random.RandomState([2012,11,1])

    mean = rng.uniform(1e-6, 1. - 1e-6, (n,))

    z = inverse_sigmoid_numpy(mean)

    layer.set_biases(z.astype(config.floatX))

    init_state = layer.make_state(num_examples=num_samples,
            numpy_rng=rng)

    value = init_state.get_value()

    check_binary_samples(value, (num_samples, n), mean, tol)

def test_binary_vis_layer_sample():

    # Verifies that BinaryVector.sample returns an expression
    # whose value passes check_binary_samples

    assert hasattr(np, 'exp')

    n = 5
    num_samples = 1000
    tol = .04

    class DummyLayer(object):
        """
        A layer that we build for the test that just uses a state
        as its downward message.
        """

        def downward_state(self, state):
            return state

        def downward_message(self, state):
            return state

    vis = BinaryVector(nvis=n)
    hid = DummyLayer()

    rng = np.random.RandomState([2012,11,1,259])

    mean = rng.uniform(1e-6, 1. - 1e-6, (n,))

    ofs = rng.randn(n)

    vis.set_biases(ofs.astype(config.floatX))

    z = inverse_sigmoid_numpy(mean) - ofs

    z_var = sharedX(np.zeros((num_samples, n)) + z)

    theano_rng = MRG_RandomStreams(2012+11+1)

    sample = vis.sample(state_above=z_var, layer_above=hid,
            theano_rng=theano_rng)

    sample = sample.eval()

    check_binary_samples(sample, (num_samples, n), mean, tol)

