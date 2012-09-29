"""Test pooling-related code in pooling.py"""

import numpy as np
from pylearn2.utils.pooling import pooling_matrix
from pylearn2.testing.skip_if_no_scipy


def test_pooling_no_topology():
    mat = pooling_matrix(4, 5)
    assert mat.shape == (4, 20)
    expected = np.array([[1] * 5 + [0] * 15,
                         [0] * 5 + [1] * 5 + [0] * 10,
                         [0] * 10 + [1] * 5 + [0] * 5,
                         [0] * 15 + [1] * 5])
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix(4, 5, sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_pooling_1d_topology():
    mat = pooling_matrix(3, 4, 2)
    assert mat.shape == (3, 8)
    expected = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1]])
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix(3, 4, 2, sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_pooling_1d_topology_tuples():
    mat = pooling_matrix((3,), (4,), (2,))
    assert mat.shape == (3, 8)
    expected = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1]])
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix(3, 4, 2, sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_pooling_2d_topology():
    mat = pooling_matrix((3, 3), (2, 2), (1, 1))
    assert mat.shape == (9, 16)
    expected = np.zeros((9, 16))
    maps = expected.reshape((3, 3, 4, 4))
    maps[0, 0, 0:2, 0:2] = 1.
    maps[1, 0, 1:3, 0:2] = 1.
    maps[2, 0, 2:4, 0:2] = 1.
    maps[0, 1, 0:2, 1:3] = 1.
    maps[1, 1, 1:3, 1:3] = 1.
    maps[2, 1, 2:4, 1:3] = 1.
    maps[0, 2, 0:2, 2:4] = 1.
    maps[1, 2, 1:3, 2:4] = 1.
    maps[2, 2, 2:4, 2:4] = 1.
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix((3, 3), (2, 2), (1, 1), sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_pooling_2d_topology_stride2():
    mat = pooling_matrix((3, 3), (3, 3), (2, 2))
    expected = np.zeros((9, 49))
    maps = expected.reshape((3, 3, 7, 7))
    maps[0, 0, 0:3, 0:3] = 1.
    maps[0, 1, 0:3, 2:5] = 1.
    maps[0, 2, 0:3, 4:7] = 1.
    maps[1, 0, 2:5, 0:3] = 1.
    maps[1, 1, 2:5, 2:5] = 1.
    maps[1, 2, 2:5, 4:7] = 1.
    maps[2, 0, 4:7, 0:3] = 1.
    maps[2, 1, 4:7, 2:5] = 1.
    maps[2, 2, 4:7, 4:7] = 1.
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix((3, 3), (3, 3), (2, 2), sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_pooling_2d_non_overlapping():
    mat = pooling_matrix((3, 3), (3, 3), (3, 3))
    assert mat.shape == (9, 81)
    expected = np.zeros((9, 81))
    maps = expected.reshape((3, 3, 9, 9))
    maps[0, 0, 0:3, 0:3] = 1.
    maps[0, 1, 0:3, 3:6] = 1.
    maps[0, 2, 0:3, 6:9] = 1.
    maps[1, 0, 3:6, 0:3] = 1.
    maps[1, 1, 3:6, 3:6] = 1.
    maps[1, 2, 3:6, 6:9] = 1.
    maps[2, 0, 6:9, 0:3] = 1.
    maps[2, 1, 6:9, 3:6] = 1.
    maps[2, 2, 6:9, 6:9] = 1.
    assert np.all(mat == expected)
    skip_if_no_scipy()
    spmat = pooling_matrix((3, 3), (3, 3), (3, 3), sparse='csr')
    assert np.all(spmat.todense() == expected)


def test_exceptions():
    def check_raised(exc_type, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except exc_type:
            return
        assert False

    yield (check_raised, TypeError, pooling_matrix, 'hello', 2)
    yield (check_raised, TypeError, pooling_matrix, 2, 'hello')
    yield (check_raised, TypeError, pooling_matrix, 3, 3, 'hello')
    yield (check_raised, ValueError, pooling_matrix, 2, (3, 4))
    yield (check_raised, ValueError, pooling_matrix, (4, 5), 2)
    yield (check_raised, ValueError, pooling_matrix, (4, 5), (5, 6), (3,))
    yield (check_raised, ValueError, pooling_matrix, (4, 5, 6), (3, 2))
    yield (check_raised, ValueError, pooling_matrix, (3, 3), (2, 2), (5, 5))
    yield (check_raised, ValueError, pooling_matrix,
           (3, 3, 3), (2, 2, 2), (2, 2, 1))
    yield (check_raised, ValueError, pooling_matrix, 5, 2, 1, 'float32', 'abc')
