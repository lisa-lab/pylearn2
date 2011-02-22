"""Optimizers tell models how to update their parameters during learning."""
# Third-party imports
import theano
from theano import tensor
from pylearn.gd.sgd import sgd_updates

# Local imports
from base import Optimizer
from utils import safe_update, sharedX

floatX = theano.config.floatX

class SGDOptimizer(Optimizer):
    """
    Compute updates by stochastic gradient descent on mini-batches.

    Supports constant learning rates, or decreasing like 1/t after an initial
    period.
    """

    def __init__(self, conf, params, cost):
        """
        :type conf: Dictionary
        :param conf: Other configuration variables. Are supported:
            * base_lr: the base learning rate
            * <paramname>_lr: specific modifier of the learning rate applied
              on parameter <paramname>. Defaults to 1.
            * lr_anneal_start: Annealing coefficient.

        :type params: Either a list of shared variables, or an object with
            a 'params()' method returning such a list.
        :param params: The parameters we want to update.

        :type cost: A symbolic Theano variable.
        :param cost: The cost to minimize.

        The formula to compute the effective learning rate on a parameter is:
        <paramname>_lr * min(0.0, max(base_lr, lr_anneal_start/(iteration+1)))
        """
        if isinstance(params, (list, tuple)):
            self.params = params
        else:
            self.params = params.params()
        self.cost = cost
        self.conf = conf
        self.learning_rates_setup(conf, params)

    def updates(self):
        """Compute the updates for each of the parameter variables."""
        ups = {}
        # Get the gradient w.r.t. cost of each parameter.
        l_ups, learn_rates = self.learning_rate_updates(self.conf, self.params)

        # Add the learning rate/iteration updates
        safe_update(ups, l_ups)
        grads = [
            tensor.grad(self.cost, p)
            for p in self.params
        ]
        # Get the updates from sgd_updates, a PyLearn library function.
        p_up = dict(sgd_updates(self.params, grads, learn_rates))

        # Add the things in p_up to ups
        safe_update(ups, p_up)

        # Return the updates dictionary.
        return ups

    def function(self, inputs, name=None):
        """Compile the Theano training function associated with the optimizer"""
        return theano.function(
                inputs,
                self.cost,
                updates=self.updates(),
                name=name)
