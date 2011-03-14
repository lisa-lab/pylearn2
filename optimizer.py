"""Optimizers tell models how to update their parameters during learning."""
# Third-party imports
from numpy import inf
import theano
from theano import tensor
from pylearn.gd.sgd import sgd_updates

# Local imports
from .base import Optimizer
from .utils import safe_update, sharedX

floatX = theano.config.floatX

class SGDOptimizer(Optimizer):
    """
    Compute updates by stochastic gradient descent on mini-batches.

    Supports constant learning rates, or decreasing like 1/t after an initial
    period.
    """

    def __init__(self, params, base_lr, anneal_start=None, **kwargs):
        """
        :param base_lr: the base learning rate
        :param <paramname>_lr: specific modifier of the learning rate applied
              on parameter <paramname>. Defaults to 1.
        :param anneal_start: Annealing coefficient.

        :type params: Either a list of shared variables, or an object with
            a 'params()' method returning such a list.
        :param params: The parameters we want to update.

        The formula to compute the effective learning rate on a parameter is:
        <paramname>_lr * min(0.0, max(base_lr, lr_anneal_start/(iteration+1)))
        """
        if hasattr(params, '__iter__'):
            self.params = params
        else:
            self.params = params.params()
        if anneal_start == None:
            self.anneal_start = inf
        else:
            self.anneal_start = tensor.cast(anneal_start, floatX)
        self.learning_rates_setup(base_lr, **kwargs)

    def learning_rates_setup(self, base_lr, **kwargs):
        """
        Initializes parameter-specific learning rate dictionary and shared
        variables for the annealed base learning rate and iteration number.
        """
        # Take care of learning rate scales for individual parameters
        self.learning_rates = {}
        # Base learning rate per example.
        self.base_lr = theano._asarray(base_lr, dtype=floatX)

        for parameter in self.params:
            lr_name = '%s_lr' % parameter.name
            thislr = kwargs.get(lr_name, 1.)
            self.learning_rates[parameter] = sharedX(thislr, lr_name)

        # A shared variable for storing the iteration number.
        self.iteration = sharedX(theano._asarray(0, dtype='int32'),
                                 name='iter')

        # A shared variable for storing the annealed base learning rate, used
        # to lower the learning rate gradually after a certain amount of time.
        self.annealed = sharedX(base_lr, 'annealed')

    def learning_rate_updates(self):
        ups = {}

        # Annealing coefficient. Here we're using a formula of
        # base_lr * min(0.0, max(base_lr, anneal_start / (iteration + 1))
        frac = self.anneal_start / (self.iteration + 1.)
        annealed = tensor.clip(
            tensor.cast(frac, floatX),
            0.0,    # minimum learning rate
            self.base_lr # maximum learning rate
        )

        # Update the shared variable for the annealed learning rate.
        ups[self.annealed] = annealed
        ups[self.iteration] = self.iteration + 1

        # Calculate the learning rates for each parameter, in the order
        # they appear in self.params
        learn_rates = [annealed * self.learning_rates[p] for p in self.params]
        return ups, learn_rates

    def updates(self, gradients):
        """Return symbolic updates to apply.

        The updates are computed to follow the gradient of a cost
        (or pseudo-cost), wrt self.parameters.

        :type gradients: A list of symbolic Theano variables, the same
        length as self.model
        :param gradients: The gradients of a cost (or pseudo-cost) wrt
        self.params.
        """
        ups = {}
        # Add the learning rate/iteration updates
        l_ups, learn_rates = self.learning_rate_updates()
        safe_update(ups, l_ups)

        # Get the updates from sgd_updates, a PyLearn library function.
        p_up = dict(sgd_updates(self.params, gradients, learn_rates))

        # Add the things in p_up to ups
        safe_update(ups, p_up)

        # Return the updates dictionary.
        return ups

    def cost_updates(self, cost):
        """Return symbolic updates given a cost to minimize

        :type cost: A scalar symbolic Theano variable
        :param cost: The cost to minimize.

        self.cost_updates(cost) is equivalent to
        self.updates(T.grad(cost, self.params))
        """
        grads = [tensor.grad(cost, p) for p in self.params]
        return self.updates(gradients=grads)

    def ml_updates(self, model, sampler, visible_batch):
        """Compute the updates given an estimator of the likelihood

        TODO: document args
        """
        pos_v = visible_batch
        neg_v = sampler.particles
        grads = model.ml_gradients(pos_v, neg_v)
        ups = self.updates(gradients=grads)

        # Add the sampler's updates (negative phase particles, etc.).
        safe_update(ups, sampler.updates())

        return ups

