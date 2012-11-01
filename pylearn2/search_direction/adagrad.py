import theano
import numpy
from pylearn2.search_direction.search_direction import SearchDirection

class Adagrad(SearchDirection):
    def __init__(self, c):
        self.c = c
    
    def dir_from_grad(self, gradients):
        updates = {}
        direction = {}
        for param, grad in gradients.iteritems():
            sqr_grad_sum = theano.shared(numpy.zeros_like(param.get_value()))
            updates[sqr_grad_sum] = sqr_grad_sum + grad**2
            direction[param] = theano.tensor.cast(grad / theano.tensor.sqrt(sqr_grad_sum + self.c), 'float32')

        return direction, updates