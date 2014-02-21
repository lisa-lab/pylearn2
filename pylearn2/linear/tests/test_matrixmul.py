from pylearn2.linear.matrixmul import MatrixMul
import pdb
import theano
from theano import tensor
import numpy as np

def test_matrixmul():
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
            np_W.append(np.random.randint(
                -10, 10, np.random.random_integers(5, size=2)
            ).astype(dtype))
            np_x.append(np.random.randint(
                -10, 10, (np.random.random_integers(5),
                          np_W[-1].shape[0])
            ).astype(dtype))
            np_x_T.append(np.random.randint(
                -10, 10, (np.random.random_integers(5),
                          np_W[-1].shape[1])
            ).astype(dtype))
        elif 'float' in dtype:
            np_W.append(np.random.uniform(
                -1, 1, np.random.random_integers(5, size=2)
            ).astype(dtype))
            np_x.append(np.random.uniform(
                -10, 10, (np.random.random_integers(5),
                          np_W[-1].shape[0])
            ).astype(dtype))
            np_x.append(np.random.uniform(
                -10, 10, (np.random.random_integers(5),
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
