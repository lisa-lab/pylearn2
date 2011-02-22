import numpy
import theano
from theano import tensor
from pylearn.gd.sgd import sgd_updates

from framework.base import Optimizer
from framework.utils import safe_update, sharedX

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

        # Take care of learning rate scales for individual parameters
        self.learning_rates = {}

        for parameter in self.params:
            lr_name = '%s_lr' % parameter.name
            thislr = conf.get(lr_name, 1.)
            self.learning_rates[parameter] = sharedX(thislr, lr_name)

        # A shared variable for storing the iteration number.
        self.iteration = sharedX(theano._asarray(0, dtype='int32'), name='iter')

        # A shared variable for storing the annealed base learning rate, used
        # to lower the learning rate gradually after a certain amount of time.
        self.annealed = sharedX(conf['base_lr'], 'annealed')

    def updates(self):
        """Compute the updates for each of the parameter variables."""
        ups = {}
        # Base learning rate per example.
        base_lr = theano._asarray(self.conf['base_lr'], dtype=floatX)

        # Annealing coefficient. Here we're using a formula of
        # base_lr * min(0.0, max(base_lr, lr_anneal_start / (iteration + 1))
        frac = self.conf['lr_anneal_start'] / (self.iteration + 1.)
        annealed = tensor.clip(
            tensor.cast(frac, floatX),
            0.0,    # minimum learning rate
            base_lr # maximum learning rate
        )

        # Update the shared variable for the annealed learning rate.
        ups[self.annealed] = annealed
        ups[self.iteration] = self.iteration + 1

        # Calculate the learning rates for each parameter, in the order
        # they appear in self.params
        learn_rates = [annealed * self.learning_rates[p] for p in self.params]
        # Get the gradient w.r.t. cost of each parameter.
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

