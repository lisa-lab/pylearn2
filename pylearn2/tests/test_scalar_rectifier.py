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
relu__ = lambda x: T.switch(x < 0., 0., x)

def test_scalar_rectifier():
    # verify the new op rectifier produces the same results as relu
    x = T.fmatrix('inputs')
    y1 = relu(x)
    y2 = rectifier(x)
    y3 = relu_(x)
    y4 = relu__(x)
    
    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_1_forward')
    f2 = theano.function(inputs=[x], outputs=y2, name='benchmark_2_forward')
    f3 = theano.function(inputs=[x], outputs=y3, name='benchmark_3_forward')
    f4 = theano.function(inputs=[x], outputs=y4, name='benchmark_4_forward')
    
    g1 = theano.function(inputs=[x], outputs=T.grad(y1.sum(),x), name='benchmark_1_grad')
    g2 = theano.function(inputs=[x], outputs=T.grad(y2.sum(),x), name='benchmark_2_grad')
    g3 = theano.function(inputs=[x], outputs=T.grad(y3.sum(),x), name='benchmark_3_grad')
    g4 = theano.function(inputs=[x], outputs=T.grad(y4.sum(),x), name='benchmark_4_grad')
    
    for i in range(10):
        value = numpy.random.uniform(-1,1,size=(100,500)).astype(floatX)
        numpy.testing.assert_array_equal(f1(value), f2(value),
                                         err_msg='arrays not equal' )

        numpy.testing.assert_array_equal(f1(value), f3(value),
                                         err_msg='arrays not equal' )

        numpy.testing.assert_array_equal(f1(value), f4(value),
                                         err_msg='arrays not equal' )

        numpy.testing.assert_array_equal(g1(value), g2(value),
                                         err_msg='grad:arrays not equal' )

        numpy.testing.assert_array_equal(g1(value), g3(value),
                                         err_msg='grad:arrays not equal' )
        
        numpy.testing.assert_array_equal(g1(value), g4(value),
                                         err_msg='grad:arrays not equal' )


def benchmark_single_op():
    x = T.ftensor4('inputs')
    
    ops = [
        rectifier(x).sum(), # new
        relu_(x).sum(), # old
        relu(x).sum(), # alter, short for alternative
        relu__(x).sum(), # alter 2
        T.grad(rectifier(x).sum(),x), # grad_new
        T.grad(relu_(x).sum(),x), # grad_old
        T.grad(relu(x).sum(),x), # grad_alter
        T.grad(relu__(x).sum(),x), # grad_alter2
    ]

    names = ['fprop_new', 'fprop_old', 'fprop_alter', 'fprop_alter2',
             'grad_new', 'grad_old', 'grad_alter', 'grad_alter2']

    
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
        print name
        theano.printing.debugprint(f, print_type=True)
    print names
    print times
            
def benchmark_all():
    benchmark_single_op()

if __name__ == '__main__':
    benchmark_all()
    #test_scalar_rectifier()