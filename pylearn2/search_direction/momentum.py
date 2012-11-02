import theano
import numpy
from pylearn2.utils import sharedX
from pylearn2.search_direction.search_direction import SearchDirection


class Momentum(SearchDirection):
    def __init__(self, mass):
        self.mass = mass
    
    def dir_from_grad(self, gradients):
        updates = {}
        direction = {}
        for param, grad in gradients.iteritems():
            v = sharedX(numpy.zeros_like(param.get_value()))
            assert v.dtype == grad.dtype
            updates[v] = grad - self.mass * v
            direction[param] = v

        return direction, updates