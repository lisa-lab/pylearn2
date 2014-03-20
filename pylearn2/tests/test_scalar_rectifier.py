import theano
import theano.tensor as T
from pylearn2.scalar import rectifier
import numpy
import time

floatX = 'float32'
relu = lambda x: T.maximum(0.0, x)

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
    '''
    On CPU, the new scalar_rectifier is about 1.5 times faster.
    On GPU, they are almost of the same speed. 
    '''
    x = T.fmatrix('inputs')
    y1 = relu(x).sum()
    y2 = rectifier(x).sum()

    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_1')
    f2 = theano.function(inputs=[x], outputs=y2, name='benchmark_2')

    n_loops = 100
    value = numpy.random.uniform(size=(100,5000)).astype(floatX)
    
    t0 = time.time()
    for i in range(n_loops):
        f1(value)
    t1 = time.time()
    benchmark_1 = t1-t0
    
    t0 = time.time()
    for i in range(n_loops):
        f2(value)
    t1 = time.time()
    benchmark_2 = t1-t0

    print 'speed benchmark relu/scalar_rectifier= ', benchmark_1/(benchmark_2+0.0)
    
def benchmark_all():
    benchmark_single_op()

if __name__ == '__main__':
    #benchmark_all()
    test_scalar_rectifier()