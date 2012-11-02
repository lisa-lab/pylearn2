import theano
from pylearn2.utils import sharedX
from pylearn2.search_direction.search_direction import SearchDirection


class LRDecay(SearchDirection):
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate

    def dir_from_grad(self, gradients):
        time = sharedX(0)
        decay = 1. / (1. + self.decay_rate * time)

        updates = {}
        updates[time] = time + 1
        direction = {}
        for param, grad in gradients.iteritems():
            direct = grad * decay
            assert direct.dtype == grad.dtype
            direction[param] = grad * decay

        return direction, updates
        