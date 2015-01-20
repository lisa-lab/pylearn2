'''
This is the benchmark of 4 different implementations
of rectified linear activation in Theano.
Two types of computations are tested w.r.t.
each implementation: fprop and grad.

Results: in seconds, float32 (details in the code)

Implementation, CPU (fprop, bprop), GPU (fprop, bprop), (final score)
a) ScalarRectifier:       (2.32, 2.40)    (1.36, 2.67)    (8.75)
b) T.max(.0, x):          (5.19, 3.65)    (1.38, 2.38)    (12.60)
c) x*(x>0.):              (2.85, 2.84)    (1.31, 2.91)    (9.91)
d) T.switch(x<0., 0., x): (2.32, 1.41)    (1.41, 2.84)    (8.39)

Conlusion:
In terms of efficiency, d) > a) > c) > b)
'''
from __future__ import print_function

__authors__ = "Li Yao and Frederic Bastien"

import theano
import theano.tensor as T

import numpy
import time

floatX = 'float32'
def relu(x):
    """
    relu implementation with T.maximum

    Parameters
    ----------
    x: tensor variable
    """
    return T.maximum(0.0, x)

def relu_(x):
    """
    Alternative relu implementation

    Parameters
    ----------
    x: tensor variable
    """
    return x * (x > 0)

def relu__(x):
    """
    Alternative relu implementation. The most efficient one.

    Parameters
    ----------
    x: tensor variable
    """
    return T.switch(x < 0., 0., x)


def test_scalar_rectifier():
    """
    verify different implementations of relu
    """
    x = T.fmatrix('inputs')
    y1 = relu(x)
    y3 = relu_(x)
    y4 = relu__(x)

    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_1_forward')
    f3 = theano.function(inputs=[x], outputs=y3, name='benchmark_3_forward')
    f4 = theano.function(inputs=[x], outputs=y4, name='benchmark_4_forward')

    g1 = theano.function(inputs=[x], outputs=T.grad(y1.sum(),x),
            name='benchmark_1_grad')
    g3 = theano.function(inputs=[x], outputs=T.grad(y3.sum(),x),
            name='benchmark_3_grad')
    g4 = theano.function(inputs=[x], outputs=T.grad(y4.sum(),x),
            name='benchmark_4_grad')

    for i in range(10):
        value = numpy.random.uniform(-1,1,size=(100,500)).astype(floatX)

        numpy.testing.assert_array_equal(f1(value), f3(value),
                                         err_msg='arrays not equal' )

        numpy.testing.assert_array_equal(f1(value), f4(value),
                                         err_msg='arrays not equal' )


        numpy.testing.assert_array_equal(g1(value), g3(value),
                                         err_msg='grad:arrays not equal' )

        numpy.testing.assert_array_equal(g1(value), g4(value),
                                         err_msg='grad:arrays not equal' )


def benchmark_relu():
    """
    Benchmark the speed of different relu implementations.
    Both fprop and grad are tested.
    """
    x = T.ftensor4('inputs')
    ops = [
        relu_(x).sum(), # old
        relu(x).sum(), # alter, short for alternative
        relu__(x).sum(), # alter 2
        T.grad(relu_(x).sum(),x), # grad_old
        T.grad(relu(x).sum(),x), # grad_alter
        T.grad(relu__(x).sum(),x), # grad_alter2
    ]

    names = ['fprop_old', 'fprop_alter', 'fprop_alter2',
             'grad_old', 'grad_alter', 'grad_alter2']


    value = numpy.random.uniform(size=(512,32,32,100)).astype(floatX)
    times = []
    for op, name in zip(ops, names):
        f = theano.function(inputs=[x], outputs=op, name=name)
        n_loops = 10

        t0 = time.time()
        for i in range(n_loops):
            f(value)
        t1 = time.time()
        benchmark = t1-t0
        times.append(benchmark)
        print(name)
        theano.printing.debugprint(f, print_type=True)
    print(names)
    print(times)


if __name__ == '__main__':
    benchmark_relu()
    #test_scalar_rectifier()
