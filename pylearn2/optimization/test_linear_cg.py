import theano
from theano import tensor
import numpy
import linear_cg
import linear_conj_grad_r
import scipy
import scipy.linalg
import time

def test_linear_cg():
    rng = numpy.random.RandomState([1,2,3])
    n = 5
    M = rng.randn(2*n,n)
    M = numpy.dot(M.T,M)
    b = rng.randn(n)
    c = rng.randn()
    x = theano.shared(rng.randn(n))
    f = 0.5 * tensor.dot(x,tensor.dot(M,x)) - tensor.dot(b,x) + c
    sol = linear_cg.linear_cg(f,[x])

    fn_sol = theano.function([], sol)

    start = time.time()
    sol  = fn_sol()[0]
    my_lcg = time.time() -start

    start = time.time()
    linear_conj_grad_r.linear_conj_grad_r(f,[x])
    ian_lcg = time.time() -start

    eval_f = theano.function([],f)
    print 'Ian verions value of f: '+str(eval_f()), 'time (s)', ian_lcg
    x.set_value(sol)
    print 'My version value of f:', str(eval_f()), 'time (s)', my_lcg
    x.set_value(   scipy.linalg.solve(M,b) , borrow = True )
    print 'true minimum value: '+str(eval_f())
if __name__ == '__main__':
    test_linear_cg()
