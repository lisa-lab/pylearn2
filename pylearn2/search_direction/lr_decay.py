import theano
from pylearn2.search_direction.search_direction import SearchDirection


class LRDecay(SearchDirection):
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate

    def dir_from_grad(self, gradients):
        time = theano.shared(0, name='time')
        decay = 1. / (1. + self.decay_rate * time)

        updates = {}
        updates[time] = time + 1
        direction = {}
        for param, grad in gradients.iteritems():
            direction[param] = theano.tensor.cast(grad * decay, 'float32')

        return direction, updates
        