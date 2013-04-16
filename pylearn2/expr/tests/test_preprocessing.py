import numpy
from pylearn2.expr.preprocessing import global_contrast_normalize


def test_basic():
    rng = numpy.random.RandomState(0)
    X = abs(rng.randn(50, 70))
    Y = global_contrast_normalize(X)
    numpy.testing.assert_allclose((Y ** 2).sum(axis=1), 1)
    numpy.testing.assert_allclose(Y.mean(axis=1), 0, atol=1e-10)


def test_scale():
    rng = numpy.random.RandomState(0)
    X = abs(rng.randn(50, 70))
    Y = global_contrast_normalize(X, scale=5)
    numpy.testing.assert_allclose(numpy.sqrt((Y ** 2).sum(axis=1)), 5)
    numpy.testing.assert_allclose(Y.mean(axis=1), 0, atol=1e-10)


def test_subtract_mean_false():
    rng = numpy.random.RandomState(0)
    X = abs(rng.randn(50, 70))
    Y = global_contrast_normalize(X, subtract_mean=False, scale=5)
    numpy.testing.assert_allclose(numpy.sqrt((Y ** 2).sum(axis=1)), 5)
    numpy.testing.assert_raises(AssertionError,
                                numpy.testing.assert_allclose,
                                Y.mean(axis=1), 0, atol=1e-10)


def test_std_norm():
    rng = numpy.random.RandomState(0)
    X = abs(rng.randn(50, 70))
    Y = global_contrast_normalize(X, use_std=True, scale=5)
    numpy.testing.assert_allclose(Y.std(axis=1, ddof=1), 5)


def test_min_divisor():
    rng = numpy.random.RandomState(0)
    X = abs(rng.randn(50, 70))
    X[0] *= 1e-15
    Y = global_contrast_normalize(X, subtract_mean=False, use_std=True)
    numpy.testing.assert_array_equal(X[0], Y[0])
