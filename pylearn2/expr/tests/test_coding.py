from pylearn2.expr.coding import triangle_code
import numpy as np
import theano.tensor as T
from theano import function
from pylearn2.utils import as_floatX

def test_triangle_code():
    rng = np.random.RandomState([20,18,9])

    m = 5
    n = 6
    k = 7

    X = as_floatX(rng.randn(m,n))
    D = as_floatX(rng.randn(k,n))

    D_norm_squared = np.sum(D**2,axis=1)
    X_norm_squared = np.sum(X**2,axis=1)
    sq_distance = -2.0 * np.dot(X,D.T) + D_norm_squared + np.atleast_2d(X_norm_squared).T
    distance = np.sqrt(sq_distance)

    mu = np.mean(distance, axis = 1)
    expected = np.maximum(0.0,mu.reshape(mu.size,1)-distance)

    Xv = T.matrix()
    Dv = T.matrix()

    code = triangle_code(X = Xv, centroids = Dv)
    actual = function([Xv,Dv],code)(X,D)

    assert np.allclose(expected, actual)
