import theano
import numpy
from pylearn2.utils import sharedX
from pylearn2.search_direction.search_direction import SearchDirection


class Momentum(SearchDirection):
    """
    A search direction implementation of momentum.
    """
    def __init__(self, mass, learning_rate):
        """
        Normally the momentum is computed as follows:
            v := mass * v - learning_rate * grad(cost, param)
        But to comply with SGD, which expects a single search direction to
        compute
            param := param - learning_rate * search_direction
        we need to factor out -learning_rate, which gives us
            v := grad(cost, param) - (mass / learning_rate) * v

        It is therefore VERY IMPORTANT that the learning rate passed to
        Momentum is the same theano shared variable as the one passed to SGD.
        """
        self.mass = mass
        self.learning_rate = learning_rate

    def dir_from_grad(self, gradients):
        """
        The values computed here might seem wrong for momentum, but they're
        simply refactored to comply with the way SGD works.
        
        Since SGD computes the updates as
            param_t := param_t - learning_rate * search_direction_t,
        if we define search_direction as
            search_direction_t = grad_{t-1} - (mass / learning_rate) * v_{t-1}
        and v as
            v_t := param_{t-1} + (mass * v_{t-1})
                               - (learning_rate * grad_{t-1})
        we get updates in the form
            param_t := param_{t-1} - learning_rate * (grad{t-1}
                                   - (mass / learning_rate) * v_{t-1})
                     = param_{t-1} + (mass * v_{t-1})
                                   - (learning_rate * grad_{t-1})
                     = param_{t-1} + v_t,
        which is what we expect to have for momentum.
        """
        updates = {}
        direction = {}
        for param, grad in gradients.iteritems():
            v = sharedX(numpy.zeros_like(param.get_value()))
            assert v.dtype == grad.dtype
            updates[v] = self.mass * v - self.learning_rate * grad
            direction[param] = grad - (self.mass / self.learning_rate) * v

        return direction, updates
