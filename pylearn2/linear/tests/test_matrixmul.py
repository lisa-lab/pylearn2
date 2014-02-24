from pylearn2.linear.matrixmul import MatrixMul, make_local_rfs
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import theano
from theano import tensor
import numpy as np


def test_matrixmul():
    """
    Tests matrix multiplication for a range of different
    dtypes. Checks both normal and transpose multiplication
    using randomly generated matrices.
    """
    rng = np.random.RandomState(222)
    dtypes = [
        'int16', 'int32', 'int64', 'float64', 'float32'
    ]
    tensor_x = [
        tensor.wmatrix(),
        tensor.imatrix(),
        tensor.lmatrix(),
        tensor.dmatrix(),
        tensor.fmatrix()
    ]
    np_W, np_x, np_x_T = [], [], []
    for dtype in dtypes:
        if 'int' in dtype:
            np_W.append(rng.randint(
                -10, 10, rng.random_integers(5, size=2)
            ).astype(dtype))
            np_x.append(rng.randint(
                -10, 10, (rng.random_integers(5),
                          np_W[-1].shape[0])
            ).astype(dtype))
            np_x_T.append(rng.randint(
                -10, 10, (rng.random_integers(5),
                          np_W[-1].shape[1])
            ).astype(dtype))
        elif 'float' in dtype:
            np_W.append(rng.uniform(
                -1, 1, rng.random_integers(5, size=2)
            ).astype(dtype))
            np_x.append(rng.uniform(
                -10, 10, (rng.random_integers(5),
                          np_W[-1].shape[0])
            ).astype(dtype))
            np_x.append(rng.uniform(
                -10, 10, (rng.random_integers(5),
                          np_W[-1].shape[1])
            ).astype(dtype))
        else:
            assert False

    def sharedW(value, dtype):
        return theano.shared(theano._asarray(value, dtype=dtype))
    tensor_W = [sharedW(W, dtype) for W in np_W]
    matrixmul = [MatrixMul(W) for W in tensor_W]
    assert all(mm.get_params()[0] == W for mm, W in zip(matrixmul, tensor_W))

    fn = [theano.function([x], mm.lmul(x))
          for x, mm in zip(tensor_x, matrixmul)]
    fn_T = [theano.function([x], mm.lmul_T(x))
            for x, mm in zip(tensor_x, matrixmul)]
    for W, x, x_T, f, f_T in zip(np_W, np_x, np_x_T, fn, fn_T):
        np.testing.assert_allclose(f(x), np.dot(x, W))
        np.testing.assert_allclose(f_T(x_T), np.dot(x_T, W.T))


def test_make_local_rfs():
    view_converter = DefaultViewConverter((10, 10, 3))
    test_dataset = DenseDesignMatrix(np.ones((10, 300)),
                                     view_converter=view_converter)
    matrixmul = make_local_rfs(test_dataset, 4, (5, 5), (5, 5),
                               draw_patches=True)
    W = matrixmul.get_params()[0].get_value()
    assert W.shape == (300, 4)
    np.testing.assert_allclose(W.sum(axis=0), 75 * np.ones(4))
    np.testing.assert_allclose(W.sum(axis=1), np.ones(300))

    matrixmul = make_local_rfs(test_dataset, 4, (5, 5), (5, 5))
    W = matrixmul.get_params()[0].get_value()
    assert W.shape == (300, 4)
    np.testing.assert_raises(ValueError, make_local_rfs,
                             test_dataset, 2, (5, 5), (5, 5))
