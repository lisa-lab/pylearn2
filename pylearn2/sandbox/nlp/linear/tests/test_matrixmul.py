"""
Sandbox tests for the projection operator
"""
import numpy as np
import theano

from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.utils import sharedX
from theano import tensor


def test_matrixmul():
    """
    Tests for projection
    """
    rng = np.random.RandomState(222)
    dtypes = [
        'int16', 'int32', 'int64'
    ]
    tensor_x = [
        tensor.wmatrix(),
        tensor.imatrix(),
        tensor.lmatrix(),
        tensor.wvector(),
        tensor.ivector(),
        tensor.lvector()
    ]
    np_W, np_x = [], []
    for dtype in dtypes:
        np_W.append(rng.rand(10, np.random.randint(1, 10)))
        np_x.append(rng.randint(
            0, 10, (rng.random_integers(5),
                    rng.random_integers(5))
        ).astype(dtype))
    for dtype in dtypes:
        np_W.append(rng.rand(10, np.random.randint(1, 10)))
        np_x.append(
            rng.randint(0, 10, (rng.random_integers(5),)).astype(dtype)
        )

    tensor_W = [sharedX(W) for W in np_W]
    matrixmul = [MatrixMul(W) for W in tensor_W]
    assert all(mm.get_params()[0] == W for mm, W in zip(matrixmul, tensor_W))

    fn = [theano.function([x], mm.project(x))
          for x, mm in zip(tensor_x, matrixmul)]
    for W, x, f in zip(np_W, np_x, fn):
        W_x = W[x]
        if x.ndim == 2:
            W_x = W_x.reshape((W_x.shape[0], np.prod(W_x.shape[1:])))
        else:
            W_x = W_x.flatten()
        np.testing.assert_allclose(f(x), W_x)
