'''
This file has two purposes:
1. test pylearn2.scalar module (conducted in test_scalar_rectifier())
2. speed benchmark on pylearn2.scalar on CPU and GPU (conducted in benchmark_single_op)

Conclusion:
1. For pylearn2.scalar, both 'grad()' and 'c_code()' work as expected.
2. On CPU,
speed benchmark fprop old/new=  0.615451241318,
speed benchmark grad old/new=  2.8991003942

'''
import theano
import theano.tensor as T
from pylearn2.scalar import rectifier
import numpy
import time

floatX = 'float32'
relu = lambda x: T.maximum(0.0, x)
relu_ = lambda x: x * (x > 0)

def test_scalar_rectifier():
    # verify the new op rectifier produces the same results as relu
    x = T.fmatrix('inputs')
    y1 = relu(x)
    y2 = rectifier(x)

    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_1_forward')
    f2 = theano.function(inputs=[x], outputs=y2, name='benchmark_2_forward')

    g1 = theano.function(inputs=[x], outputs=T.grad(y1.sum(),x), name='benchmark_1_grad')
    g2 = theano.function(inputs=[x], outputs=T.grad(y2.sum(),x), name='benchmark_2_grad')
    
    for i in range(10):
        value = numpy.random.uniform(size=(100,500)).astype(floatX)
        numpy.testing.assert_array_equal(f1(value), f2(value),
                                         err_msg='arrays not equal' )

        numpy.testing.assert_array_equal(g1(value), g2(value),
                                         err_msg='grad:arrays not equal' )
        

def benchmark_single_op():
    x = T.fmatrix('inputs')
    
    ops = [
        rectifier(x).sum(), # new
        relu_(x).sum(), # old
        relu(x).sum(), # alter, short for alternative
        T.grad(rectifier(x).sum(),x), # grad_new
        T.grad(relu_(x).sum(),x), # grad_old
        T.grad(relu(x).sum(),x) # grad_alter
    ]

    names = ['fprop_new', 'fprop_old', 'fprop_alter',
             'grad_new', 'grad_old', 'grad_alter']

    
    times = []
    for op, name in zip(ops, names):
        f = theano.function(inputs=[x], outputs=op, name=name)
        n_loops = 1000
        value = numpy.random.uniform(size=(100,5000)).astype(floatX)

        t0 = time.time()
        for i in range(n_loops):
            f(value)
        t1 = time.time()
        benchmark = t1-t0
        times.append(benchmark)

def benchmark_all():
    benchmark_single_op()

if __name__ == '__main__':
    benchmark_all()
    #test_scalar_rectifier()