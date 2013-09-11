import numpy as np

from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.utils import sharedX

class LearningRule():

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        raise NotImplementedError()

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        raise NotImplementedError()


class Momentum(LearningRule):
    """
    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc
    """

    def __init__(self, init_momentum):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        monitor.add_channel(
                name='momentum',
                ipt=None,
                val=self.momentum,
                data_specs=(NullSpace(), ''),
                dataset=monitoring_dataset)

    def get_updates(self, learning_rate, grads, lr_scalers=None):

        updates = OrderedDict()

        for param in grads.keys():
            inc = sharedX(param.get_value() * 0.)
            if param.name is not None:
                inc.name = 'inc_'+param.name
            updated_inc = self.momentum * inc - learning_rate * lr_scalers.get(param, 1.) * grads[param]
            updates[inc] = updated_inc
            updates[param] = param + updated_inc

        return updates


class AdaDelta(LearningRule):

    def __init__(self, decay=0.95):
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):

        updates = OrderedDict()

        for param in grads.keys():

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)
            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = \
                    self.decay * mean_square_grad +\
                    (1 - self.decay) * T.sqr(grads[param])

            # Compute update
            epsilon = lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = \
                    self.decay * mean_square_dx + \
                    (1 - self.decay) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t
        
        return updates







