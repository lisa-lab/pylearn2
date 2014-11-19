from __future__ import print_function

import theano
from theano import tensor, config
import numpy
from pylearn2.optimization import linear_cg
import warnings
from pylearn2.testing.skip import skip_if_no_scipy
try:
    import scipy.linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
import time

def test_linear_cg():
    rng = numpy.random.RandomState([1,2,3])
    n = 5
    M = rng.randn(2*n,n)
    M = numpy.dot(M.T,M).astype(config.floatX)
    b = rng.randn(n).astype(config.floatX)
    c = rng.randn(1).astype(config.floatX)[0]
    x = theano.tensor.vector('x')
    f = 0.5 * tensor.dot(x,tensor.dot(M,x)) - tensor.dot(b,x) + c
    sol = linear_cg.linear_cg(f,[x])

    fn_sol = theano.function([x], sol)

    start = time.time()
    sol  = fn_sol( rng.randn(n).astype(config.floatX))[0]
    my_lcg = time.time() -start

    eval_f = theano.function([x],f)
    cgf = eval_f(sol)
    print("conjugate gradient's value of f:", str(cgf), 'time (s)', my_lcg)
    skip_if_no_scipy()
    spf = eval_f( scipy.linalg.solve(M,b) )
    print("scipy.linalg.solve's value of f: "+str(spf))

    abs_diff = abs(cgf - spf)
    if not (abs_diff < 1e-5):
        raise AssertionError("Expected abs_diff < 1e-5, got abs_diff of " +
                str(abs_diff))


if __name__ == '__main__':
    test_linear_cg()
