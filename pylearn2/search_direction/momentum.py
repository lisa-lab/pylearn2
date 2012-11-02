import theano
import numpy
from pylearn2.utils import sharedX
from pylearn2.search_direction.search_direction import SearchDirection


class Momentum(SearchDirection):
    def __init__(self, mass, decay_rate=0):
        self.mass = mass
        self.decay_rate = decay_rate
    
    def dir_from_grad(self, gradients):
        time = sharedX(0)
        decay = 1. / (1. + self.decay_rate * time)

        updates = {}
        direction = {}
        for param, grad in gradients.iteritems():
            v = sharedX(numpy.zeros_like(param.get_value()))
            assert v.dtype == grad.dtype
            updates[v] = decay * grad - self.mass * v
            direction[param] = v

        return direction, updates