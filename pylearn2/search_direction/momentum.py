import theano
import numpy
from pylearn2.search_direction.search_direction import SearchDirection


class Momentum(SearchDirection):
    def __init__(self, mass):
        self.mass = mass
    
    def dir_from_grad(self, gradients):
        updates = {}
        direction = {}
        for param, grad in gradients.iteritems():
            v = theano.shared(numpy.zeros_like(param.get_value()))
            updates[v] = grad - self.mass * v
            direction[param] = theano.tensor.cast(v, 'float32')

        return direction, updates