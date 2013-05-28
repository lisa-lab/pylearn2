import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises

from pylearn2.utils.one_hot import one_hot, k_hot, compressed_one_hot


def test_one_hot_basic():
    assert_equal(one_hot([1, 2]), [[0, 1, 0], [0, 0, 1]])
    assert_equal(one_hot([[1], [2], [1]], max_label=3),
                 [[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])


def test_one_hot_dtypes():
    int_dt = ['int8', 'int16', 'int32', 'int64']
    int_dt += ['u' + dt for dt in int_dt]
    float_dt = ['float64', 'float32', 'complex64', 'complex128']
    all_dt = int_dt + float_dt
    assert_(all(one_hot([5], dtype=dt).dtype == np.dtype(dt) for dt in all_dt))


def test_one_hot_out():
    out = np.empty((2, 3), dtype='uint8')
    assert_equal(one_hot([1, 2], out=out),
                 [[0, 1, 0], [0, 0, 1]])
    assert_equal(out, [[0, 1, 0], [0, 0, 1]])


def test_k_hot_basic():
    assert_equal(k_hot([[1], [2], [1]], max_label=3),
                 [[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
    assert_equal(k_hot([[1, 2], [2, 0], [1, 0]], max_label=3),
                 [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]])


def test_k_hot_dtypes():
    int_dt = ['int8', 'int16', 'int32', 'int64']
    int_dt += ['u' + dt for dt in int_dt]
    float_dt = ['float64', 'float32', 'complex64', 'complex128']
    all_dt = int_dt + float_dt
    assert_(all(k_hot([[5, 3]], dtype=dt).dtype == np.dtype(dt)
                for dt in all_dt))


def test_k_hot_out():
    out = np.empty((2, 3), dtype='uint8')
    assert_equal(k_hot([[1, 0], [2, 1]], out=out),
                 [[1, 1, 0], [0, 1, 1]])
    assert_equal(out, [[1, 1, 0], [0, 1, 1]])


def test_out_compressed_one_hot():
    out, uniq = compressed_one_hot([2, 5, 3])
    assert_equal(out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert_equal(uniq, [2, 3, 5])

    out, uniq = compressed_one_hot([2, 5])
    assert_equal(out, [[0], [1]])
    assert_equal(uniq, [2, 5])

    out, uniq = compressed_one_hot([2, 5], simplify_binary=False)
    assert_equal(out, [[1, 0], [0, 1]])
    assert_equal(uniq, [2, 5])


def test_exceptions():
    assert_raises(ValueError, one_hot, [5, 3], 4)
    assert_raises(ValueError, one_hot, [5., 3.])
    assert_raises(ValueError, one_hot, [[5, 3], [2, 4]])
    assert_raises(ValueError, k_hot, [[5, 3], [3, 4, 5]])
    assert_raises(ValueError, k_hot, [5, 3, 3, 4, 5])
    assert_raises(ValueError, one_hot, [5, 3, 3, 4, 5], None, None,
                  np.empty((3, 3)))
    assert_raises(ValueError, one_hot, [5, 3, 3, 4, 5], None, None,
                  np.empty((5, 3)))
    assert_raises(ValueError, one_hot, [5, 3, 3, 4, 5], None, 'int8',
                  np.empty((5, 3)))
