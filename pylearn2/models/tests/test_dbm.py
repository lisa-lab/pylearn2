__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np
assert hasattr(np, 'exp')

from theano import config
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

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

def make_random_basic_binary_dbm(
        rng,
        pool_size_1,
        num_vis = None,
        num_pool_1 = None,
        num_pool_2 = None,
        pool_size_2 = None
        ):
    """
    Makes a DBM with BinaryVector for the visible layer,
    and two hidden layers of type BinaryVectorMaxPool.
    The weights and biases are initialized randomly with
    somewhat large values (i.e., not what you'd want to
    use for learning)

    rng: A numpy RandomState.
    pool_size_1: The size of the pools to use in the first
                 layer.
    """

    if num_vis is None:
        num_vis = rng.randint(1,11)
    if num_pool_1 is None:
        num_pool_1 = rng.randint(1,11)
    if num_pool_2 is None:
        num_pool_2 = rng.randint(1,11)
    if pool_size_2 is None:
        pool_size_2 = rng.randint(1,6)

    num_h1 = num_pool_1 * pool_size_1
    num_h2 = num_pool_2 * pool_size_2

    v = BinaryVector(num_vis)
    v.set_biases(rng.uniform(-1., 1., (num_vis,)).astype(config.floatX))

    h1 = BinaryVectorMaxPool(
            detector_layer_dim = num_h1,
            pool_size = num_h1,
            layer_name = 'h1',
            irange = 1.)
    h1.set_biases(rng.uniform(-1., 1., (num_h1,)).astype(config.floatX))

    h2 = BinaryVectorMaxPool(
            detector_layer_dim = num_h2,
            pool_size = num_h2,
            layer_name = 'h2',
            irange = 1.)
    h2.set_biases(rng.uniform(-1., 1., (num_h2,)).astype(config.floatX))

    dbm = DBM(visible_layer = v,
            hidden_layers = [h1, h2],
            batch_size = 1,
            niter = 50)

    return dbm


def test_bvmp_mf_energy_consistent():

    # A test of the BinaryVectorMaxPool class
    # Verifies that the mean field update is consistent with
    # the energy function

    # Specifically, in a DBM consisting of (v, h1, h2), the
    # lack of intra-layer connections means that
    # P(h1|v, h2) is factorial so mf_update tells us the true
    # conditional.
    # We also know P(h1[i] | h1[-i], v)
    #  = P(h, v) / P(h[-i], v)
    #  = P(h, v) / sum_h[i] P(h, v)
    #  = exp(-E(h, v)) / sum_h[i] exp(-E(h, v))
    # So we can check that computing P(h[i] | v) with both
    # methods works the same way

    rng = np.random.RandomState([2012,11,1,613])

    def do_test(pool_size_1):

        # Make DBM and read out its pieces
        dbm = make_random_basic_binary_dbm(
                rng = rng,
                pool_size_1 = pool_size_1,
                # All these 1s are a debugging hack, remove these arguments
                num_vis = 1,
                num_pool_1 = 1,
                num_pool_2 = 1,
                pool_size_2 = 1
                )

        v = dbm.visible_layer
        h1, h2 = dbm.hidden_layers

        num_p = h1.get_output_space().dim

        # Choose which unit we will test
        p_idx = rng.randint(num_p)

        # Randomly pick a v, h1[-p_idx], and h2 to condition on
        # (Random numbers are generated via dbm.rng)
        layer_to_state = dbm.make_layer_to_state(1)
        v_state = layer_to_state[v]
        h1_state = layer_to_state[h1]
        h2_state = layer_to_state[h2]

        # Infer P(h1[i] | h2, v) using mean field
        expected_p, expected_h = h1.mf_update(
                state_below = v.upward_state(v_state),
                state_above = h2.downward_state(h2_state),
                layer_above = h2)

        expected_p = expected_p[0, p_idx]
        expected_h = expected_h[0, p_idx * pool_size : (p_idx + 1) * pool_size]

        expected_p, expected_h = function([], [expected_p, expected_h])()

        # Infer P(h1[i] | h2, v) using the energy function
        energy = dbm.energy(V = v_state,
                hidden = [h1_state, h2_state])
        unnormalized_prob = T.exp(-energy)
        unnormalized_prob = function([], unnormalized_prob)

        p_state, h_state = h1_state

        def compute_unnormalized_prob(which_detector):
            write_h = np.zeros((pool_size_1,))
            if which_detector is None:
                write_p = 0.
            else:
                write_p = 1.
                write_h[which_detector] = 1.

            h_value = h_state.get_value()
            p_value = p_state.get_value()

            h_value[0, p_idx * pool_size : (p_idx + 1) * pool_size] = write_h
            p_value[0, p_idx] = write_p

            h_state.set_value(h_value)
            p_state.set_value(p_value)

            return unnormalized_prob()

        off_prob = compute_unnormalized_prob(None)
        on_probs = [compute_unnormalized_prob(idx) for idx in xrange(pool_size)]
        denom = off_prob + sum(on_probs)
        off_prob /= denom
        on_probs = [on_prob / denom for on_prob in on_probs]
        assert np.allclose(1., off_prob + sum(on_probs))

        # np.asarray(on_probs) doesn't make a numpy vector, so I do it manually
        wtf_numpy = np.zeros((pool_size_1,))
        for i in xrange(pool_size_1):
            wtf_numpy[i] = on_probs[i]
        on_probs = wtf_numpy

        # Check that they match
        if not np.allclose(expected_p, 1. - off_prob):
            print 'mean field expectation of p:',expected_p
            print 'expectation of p based on enumerating energy function values:',1. - off_prob
            assert False
        if not np.allclose(expected_h, on_probs):
            print 'mean field expectation of h:',expected_h
            print 'expectation of h based on enumerating energy function values:',on_probs
            assert False


    for pool_size in [1, 2, 5]:
        do_test(pool_size)

