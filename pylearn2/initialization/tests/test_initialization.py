import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import theano

from pylearn2.initialization import Constant, IsotropicGaussian, Uniform
from pylearn2.initialization import SparseInitialization


def test_constant():
    def check_constant(const, shape, ground_truth):
        # rng unused, so pass None.
        init = Constant(const).initialize(None, ground_truth.shape)
        assert_(ground_truth.dtype == theano.config.floatX)
        assert_(ground_truth.shape == init.shape)
        assert_equal(ground_truth, init)

    # Test scalar init.
    yield (check_constant, 5, (5, 5),
           5 * np.ones((5, 5), dtype=theano.config.floatX))
    # Test broadcasting.
    yield (check_constant, [1, 2, 3], (7, 3),
           np.array([[1, 2, 3]] * 7, dtype=theano.config.floatX))
    yield (check_constant, np.array([[1], [2], [3]]), (3, 2),
           np.array([[1, 1], [2, 2], [3, 3]], dtype=theano.config.floatX))


def test_gaussian():
    rng = np.random.RandomState([2014, 01, 20])

    def check_gaussian(rng, mean, std, shape):
        weights = IsotropicGaussian(mean, std).initialize(rng, shape)
        assert_(weights.shape == shape)
        assert_(weights.dtype == theano.config.floatX)
        assert_allclose(weights.mean(), mean, atol=1e-2)
        assert_allclose(weights.std(), std, atol=1e-2)
    yield check_gaussian, rng, 0, 1, (500, 600)
    yield check_gaussian, rng, 5, 3, (600, 500)


def test_uniform():
    rng = np.random.RandomState([2014, 01, 20])

    def check_uniform(rng, mean, width, std, shape):
        weights = Uniform(mean=mean, width=width,
                          std=std).initialize(rng, shape)
        assert_(weights.shape == shape)
        assert_(weights.dtype == theano.config.floatX)
        assert_allclose(weights.mean(), mean, atol=1e-2)
        if width is not None:
            std_ = width / np.sqrt(12)
        else:
            std_ = std
        assert_allclose(std_, weights.std(), atol=1e-2)
    yield check_uniform, rng, 0, 0.05, None, (500, 600)
    yield check_uniform, rng, 0, None, 0.001, (600, 500)
    yield check_uniform, rng, 5, None, 0.004, (700, 300)


def test_sparse_init():
    rng = np.random.RandomState([2014, 01, 20])

    def check_sparse_init(rng, base, num_nonzero, prob_nonzero,
                          atom_axis, shape):
        sp = SparseInitialization(base, num_nonzero=num_nonzero,
                                  prob_nonzero=prob_nonzero)
        x = sp.initialize(rng, shape)
        if num_nonzero is not None:
            x_ = (x != 0).sum(axis=atom_axis)
            assert_allclose(num_nonzero * np.ones_like(x_), x_)
        else:
            assert_allclose(prob_nonzero, (x == 0).mean(), atol=1e-2)

    yield check_sparse_init, rng, IsotropicGaussian(), 4, None, -1, (5, 5)
