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
from pylearn2.models.dbm import Softmax
from pylearn2.space import VectorSpace
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

def check_bvmp_samples(value, num_samples, n, pool_size, mean, tol):
    """
    bvmp=BinaryVectorMaxPool
    value: a tuple giving (pooled batch, detector batch)   (all made with same params)
    num_samples: number of samples there should be in the batch
    n: detector layer dimension
    pool_size: size of each pool region
    mean: (expected value of pool unit, expected value of detector units)
    tol: amount the emprical mean is allowed to deviate from the analytical expectation

    checks that:
        1) all values are binary
        2) detector layer units are mutually exclusive
        3) pooled unit is max of the detector units
        4) correct number of samples is present
        5) variables are of the right shapes
        6) samples converge to the right expected value
    """

    pv, hv = value

    assert n % pool_size == 0
    num_pools = n // pool_size

    assert pv.ndim == 2
    assert pv.shape[0] == num_samples
    assert pv.shape[1] == num_pools

    assert hv.ndim == 2
    assert hv.shape[0] == num_samples
    assert hv.shape[1] == n

    assert is_binary(pv)
    assert is_binary(hv)

    for i in xrange(num_pools):
        sub_p = pv[:,i]
        assert sub_p.shape == (num_samples,)
        sub_h = hv[:,i*pool_size:(i+1)*pool_size]
        assert sub_h.shape == (num_samples, pool_size)
        if not np.all(sub_p == sub_h.max(axis=1)):
            for j in xrange(num_samples):
                print sub_p[j], sub_h[j,:]
                assert sub_p[j] == sub_h[j,:]
            assert False
        assert np.max(sub_h.sum(axis=1)) == 1

    p, h = mean
    assert p.ndim == 1
    assert h.ndim == 1
    emp_p = pv.mean(axis=0)
    emp_h = hv.mean(axis=0)

    max_diff = np.abs(p - emp_p).max()
    if max_diff > tol:
        print 'expected value of pooling units: ',p
        print 'empirical expectation: ',emp_p
        print 'maximum difference: ',max_diff
        raise ValueError("Pooling unit samples have an unlikely mean.")
    max_diff = np.abs(h - emp_h).max()
    if max_diff > tol:
        assert False

def test_bvmp_make_state():

    # Verifies that BinaryVector.make_state creates
    # a shared variable whose value passes check_binary_samples

    num_pools = 3
    num_samples = 1000
    tol = .04
    rng = np.random.RandomState([2012,11,1,9])
    # pool_size=1 is an important corner case
    for pool_size in [1, 2, 5]:
        n = num_pools * pool_size

        layer = BinaryVectorMaxPool(
                detector_layer_dim=n,
                layer_name='h',
                irange=1.,
                pool_size=pool_size)

        # This is just to placate mf_update below
        input_space = VectorSpace(1)
        class DummyDBM(object):
            def __init__(self):
                self.rng = rng
        layer.set_dbm(DummyDBM())
        layer.set_input_space(input_space)

        layer.set_biases(rng.uniform(-pool_size, 1., (n,)).astype(config.floatX))

        # To find the mean of the samples, we use mean field with an input of 0
        mean = layer.mf_update(
                state_below=T.alloc(0., 1, 1),
                state_above=None,
                layer_above=None)

        mean = function([], mean)()

        mean = [ mn[0,:] for mn in mean ]

        state = layer.make_state(num_examples=num_samples,
                numpy_rng=rng)

        value = [elem.get_value() for elem in state]

        check_bvmp_samples(value, num_samples, n, pool_size, mean, tol)


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
            pool_size = pool_size_1,
            layer_name = 'h1',
            irange = 1.)
    h1.set_biases(rng.uniform(-1., 1., (num_h1,)).astype(config.floatX))

    h2 = BinaryVectorMaxPool(
            detector_layer_dim = num_h2,
            pool_size = pool_size_2,
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

        # Debugging checks
        num_h = h1.detector_layer_dim
        assert num_p * pool_size_1 == num_h
        pv, hv = h1_state
        assert pv.get_value().shape == (1, num_p)
        assert hv.get_value().shape == (1, num_h)

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
        assert unnormalized_prob.ndim == 1
        unnormalized_prob = unnormalized_prob[0]
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
            print 'pool_size_1:',pool_size_1

            assert False
        if not np.allclose(expected_h, on_probs):
            print 'mean field expectation of h:',expected_h
            print 'expectation of h based on enumerating energy function values:',on_probs
            assert False

    # 1 is an important corner case
    # We must also run with a larger number to test the general case
    for pool_size in [1, 2, 5]:
        do_test(pool_size)

def test_bvmp_mf_sample_consistent():

    # A test of the BinaryVectorMaxPool class
    # Verifies that the mean field update is consistent with
    # the sampling function

    # Specifically, in a DBM consisting of (v, h1, h2), the
    # lack of intra-layer connections means that
    # P(h1|v, h2) is factorial so mf_update tells us the true
    # conditional.
    # We can thus use mf_update to compute the expected value
    # of a sample of h1 from v and h2, and check that samples
    # drawn using the layer's sample method convert to that
    # value.

    rng = np.random.RandomState([2012,11,1,1016])
    theano_rng = MRG_RandomStreams(2012+11+1+1036)
    num_samples = 1000
    tol = .042

    def do_test(pool_size_1):

        # Make DBM and read out its pieces
        dbm = make_random_basic_binary_dbm(
                rng = rng,
                pool_size_1 = pool_size_1,
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

        # Debugging checks
        num_h = h1.detector_layer_dim
        assert num_p * pool_size_1 == num_h
        pv, hv = h1_state
        assert pv.get_value().shape == (1, num_p)
        assert hv.get_value().shape == (1, num_h)

        # Infer P(h1[i] | h2, v) using mean field
        expected_p, expected_h = h1.mf_update(
                state_below = v.upward_state(v_state),
                state_above = h2.downward_state(h2_state),
                layer_above = h2)

        expected_p = expected_p[0, :]
        expected_h = expected_h[0, :]

        expected_p, expected_h = function([], [expected_p, expected_h])()

        # copy all the states out into a batch size of num_samples
        cause_copy = sharedX(np.zeros((num_samples,))).dimshuffle(0,'x')
        v_state = v_state[0,:] + cause_copy
        p, h = h1_state
        h1_state = (p[0,:] + cause_copy, h[0,:] + cause_copy)
        p, h = h2_state
        h2_state = (p[0,:] + cause_copy, h[0,:] + cause_copy)

        h1_samples = h1.sample(state_below = v.upward_state(v_state),
                            state_above = h2.downward_state(h2_state),
                            layer_above = h2, theano_rng = theano_rng)

        h1_samples = function([], h1_samples)()


        check_bvmp_samples(h1_samples, num_samples, num_h, pool_size, (expected_p, expected_h), tol)


    # 1 is an important corner case
    # We must also run with a larger number to test the general case
    for pool_size in [1, 2, 5]:
        do_test(pool_size)

def check_multinomial_samples(value, expected_shape, expected_mean, tol):
    """
    Tests that a matrix of multinomial samples (observations in rows, variables
        in columns)
    1) Has the right shape
    2) Is binary
    3) Has one 1 per row
    4) Converges to the right mean
    """
    assert value.shape == expected_shape
    assert is_binary(value)
    assert np.all(value.sum(axis=1) == 1)
    mean = value.mean(axis=0)
    max_error = np.abs(mean-expected_mean).max()
    if max_error > tol:
        print 'Actual mean:'
        print mean
        print 'Expected mean:'
        print expected_mean
        print 'Maximal error:', max_error
        raise ValueError("Samples don't seem to have the right mean.")

def test_softmax_make_state():

    # Verifies that BinaryVector.make_state creates
    # a shared variable whose value passes check_multinomial_samples

    n = 5
    num_samples = 1000
    tol = .04

    layer = Softmax(n_classes = n, layer_name = 'y')

    rng = np.random.RandomState([2012, 11, 1, 11])

    z = 3 * rng.randn(n)

    mean = np.exp(z)
    mean /= mean.sum()

    layer.set_biases(z.astype(config.floatX))

    state = layer.make_state(num_examples=num_samples,
            numpy_rng=rng)

    value = state.get_value()

    check_multinomial_samples(value, (num_samples, n), mean, tol)

def test_softmax_mf_energy_consistent():

    # A test of the Softmax class
    # Verifies that the mean field update is consistent with
    # the energy function

    # Since a Softmax layer contains only one random variable
    # (with n_classes possible values) the mean field assumption
    # does not impose any restriction so mf_update simply gives
    # the true expected value of h given v.
    # We also know P(h |  v)
    #  = P(h, v) / P( v)
    #  = P(h, v) / sum_h P(h, v)
    #  = exp(-E(h, v)) / sum_h exp(-E(h, v))
    # So we can check that computing P(h | v) with both
    # methods works the same way

    rng = np.random.RandomState([2012,11,1,1131])

    # Make DBM
    num_vis = rng.randint(1,11)
    n_classes = rng.randint(1, 11)

    v = BinaryVector(num_vis)
    v.set_biases(rng.uniform(-1., 1., (num_vis,)).astype(config.floatX))

    y = Softmax(
            n_classes = n_classes,
            layer_name = 'y',
            irange = 1.)
    y.set_biases(rng.uniform(-1., 1., (n_classes,)).astype(config.floatX))

    dbm = DBM(visible_layer = v,
            hidden_layers = [y],
            batch_size = 1,
            niter = 50)

    # Randomly pick a v to condition on
    # (Random numbers are generated via dbm.rng)
    layer_to_state = dbm.make_layer_to_state(1)
    v_state = layer_to_state[v]
    y_state = layer_to_state[y]

    # Infer P(y | v) using mean field
    expected_y = y.mf_update(
            state_below = v.upward_state(v_state))

    expected_y = expected_y[0, :]

    expected_y = expected_y.eval()

    # Infer P(y | v) using the energy function
    energy = dbm.energy(V = v_state,
            hidden = [y_state])
    unnormalized_prob = T.exp(-energy)
    assert unnormalized_prob.ndim == 1
    unnormalized_prob = unnormalized_prob[0]
    unnormalized_prob = function([], unnormalized_prob)

    def compute_unnormalized_prob(which):
        write_y = np.zeros((n_classes,))
        write_y[which] = 1.

        y_value = y_state.get_value()

        y_value[0, :] = write_y

        y_state.set_value(y_value)

        return unnormalized_prob()

    probs = [compute_unnormalized_prob(idx) for idx in xrange(n_classes)]
    denom = sum(probs)
    probs = [on_prob / denom for on_prob in probs]

    # np.asarray(probs) doesn't make a numpy vector, so I do it manually
    wtf_numpy = np.zeros((n_classes,))
    for i in xrange(n_classes):
        wtf_numpy[i] = probs[i]
    probs = wtf_numpy

    if not np.allclose(expected_y, probs):
        print 'mean field expectation of h:',expected_y
        print 'expectation of h based on enumerating energy function values:',probs
        assert False

def test_softmax_mf_sample_consistent():

    # A test of the Softmax class
    # Verifies that the mean field update is consistent with
    # the sampling function

    # Since a Softmax layer contains only one random variable
    # (with n_classes possible values) the mean field assumption
    # does not impose any restriction so mf_update simply gives
    # the true expected value of h given v.
    # We can thus use mf_update to compute the expected value
    # of a sample of y conditioned on v, and check that samples
    # drawn using the layer's sample method convert to that
    # value.

    rng = np.random.RandomState([2012,11,1,1154])
    theano_rng = MRG_RandomStreams(2012+11+1+1154)
    num_samples = 1000
    tol = .042

    # Make DBM
    num_vis = rng.randint(1,11)
    n_classes = rng.randint(1, 11)

    v = BinaryVector(num_vis)
    v.set_biases(rng.uniform(-1., 1., (num_vis,)).astype(config.floatX))

    y = Softmax(
            n_classes = n_classes,
            layer_name = 'y',
            irange = 1.)
    y.set_biases(rng.uniform(-1., 1., (n_classes,)).astype(config.floatX))

    dbm = DBM(visible_layer = v,
            hidden_layers = [y],
            batch_size = 1,
            niter = 50)

    # Randomly pick a v to condition on
    # (Random numbers are generated via dbm.rng)
    layer_to_state = dbm.make_layer_to_state(1)
    v_state = layer_to_state[v]
    y_state = layer_to_state[y]

    # Infer P(y | v) using mean field
    expected_y = y.mf_update(
            state_below = v.upward_state(v_state))

    expected_y = expected_y[0, :]

    expected_y = expected_y.eval()

    # copy all the states out into a batch size of num_samples
    cause_copy = sharedX(np.zeros((num_samples,))).dimshuffle(0,'x')
    v_state = v_state[0,:] + cause_copy
    y_state = y_state[0,:] + cause_copy

    y_samples = y.sample(state_below = v.upward_state(v_state), theano_rng=theano_rng)

    y_samples = function([], y_samples)()

    check_multinomial_samples(y_samples, (num_samples, n_classes), expected_y, tol)

