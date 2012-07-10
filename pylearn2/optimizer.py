"""Optimizers tell models how to update their parameters during learning."""
# System imports
import sys

# Third-party imports
import numpy
from numpy import inf
import theano
from theano import tensor

# Local imports
from pylearn2.base import Optimizer
from pylearn2.utils import as_floatX, safe_update, sharedX


class SGDOptimizer(Optimizer):
    """
    Compute updates by stochastic gradient descent on mini-batches.

    Supports constant learning rates, or decreasing like 1/t after an initial
    period.
    """
    def __init__(self, params, base_lr, anneal_start=None, use_adagrad=False,
                 ** kwargs):
        """
        Construct an SGDOptimizer.

        Parameters
        ----------
        params : object or list
            Either a Model object with a .get_params() method, or a list of
            parameters to be optimized.
        base_lr : float
            The base learning rate before annealing or parameter-specific
            scaling.
        anneal_start : int
            Number of steps after which to start annealing the learning
            rate at a 1/t schedule, where t is the number of stochastic
            gradient updates.

        Notes
        -----
        The formula to compute the effective learning rate on a parameter is:
        <paramname>_lr * max(0.0, min(base_lr, lr_anneal_start/(iteration+1)))

        Parameter-specific learning rates can be set by passing keyword
        arguments <name>_lr, where name is the .name attribute of a given
        parameter.

        Parameter-specific bounding values can be specified by passing
        keyword arguments <param>_clip, which should be a (min, max) pair.
        """
        if hasattr(params, '__iter__'):
            self.params = params
        elif hasattr(params, 'get_params') and hasattr(params.get_params, '__call__'):
            self.params = params.get_params()
        else:
            raise ValueError("SGDOptimizer couldn't figure out what to do "
                             "with first argument: '%s'" % str(params))
        if anneal_start == None:
            self.anneal_start = None
        else:
            self.anneal_start = as_floatX(anneal_start)

        # Create accumulators and epsilon0's
        self.use_adagrad = use_adagrad
        if self.use_adagrad:
            self.accumulators = {}
            self.e0s = {}
            for param in self.params:
                self.accumulators[param] = theano.shared(value=as_floatX(0.),
                                                         name='acc_%s' % param.name)
                self.e0s[param] = as_floatX(base_lr)

        # Set up the clipping values
        self.clipping_values = {}
        # Keep track of names already seen
        clip_names_seen = set()
        for parameter in self.params:
            clip_name = '%s_clip' % parameter.name
            if clip_name in kwargs:
                if clip_name in clip_names_seen:
                    print >> sys.stderr, ('Warning: In SGDOptimizer, '
                            'at least two parameters have the same name. '
                            'Both will be affected by the keyword argument '
                            '%s.' % clip_name)
                clip_names_seen.add(clip_name)
                p_min, p_max = kwargs[clip_name]
                assert p_min <= p_max
                self.clipping_values[parameter] = (p_min, p_max)

        # Check that no ..._clip keyword is being ignored
        for clip_name in clip_names_seen:
            kwargs.pop(clip_name)
        for kw in kwargs.iterkeys():
            if kw[-5:] == '_clip':
                print >> sys.stderr, ('Warning: in SGDOptimizer, '
                        'keyword argument %s will be ignored, '
                        'because no parameter was found with name %s.'
                        % (kw, kw[:-5]))

        self.learning_rates_setup(base_lr, **kwargs)

    def learning_rates_setup(self, base_lr, **kwargs):
        """
        Initializes parameter-specific learning rate dictionary and shared
        variables for the annealed base learning rate and iteration number.

        Parameters
        ----------
        base_lr : float
            The base learning rate before annealing or parameter-specific
            scaling.

        Notes
        -----
        Parameter-specific learning rates can be set by passing keyword
        arguments <name>_lr, where name is the .name attribute of a given
        parameter.
        """
        # Take care of learning rate scales for individual parameters
        self.learning_rates = {}
        # Base learning rate per example.
        self.base_lr = theano._asarray(base_lr, dtype=theano.config.floatX)

        # Keep track of names already seen
        lr_names_seen = set()
        for parameter in self.params:
            lr_name = '%s_lr' % parameter.name
            if lr_name in lr_names_seen:
                print >> sys.stderr, ('Warning: In SGDOptimizer, '
                        'at least two parameters have the same name. '
                        'Both will be affected by the keyword argument '
                        '%s.' % lr_name)
            lr_names_seen.add(parameter.name)

            thislr = kwargs.get(lr_name, 1.)
            self.learning_rates[parameter] = sharedX(thislr, lr_name)

        # Verify that no ..._lr keyword argument is ignored
        for lr_name in lr_names_seen:
            if lr_name in kwargs:
                kwargs.pop(lr_name)
        for kw in kwargs.iterkeys():
            if kw[-3:] == '_lr':
                print >> sys.stderr, ('Warning: in SGDOptimizer, '
                        'keyword argument %s will be ignored, '
                        'because no parameter was found with name %s.'
                        % (kw, kw[:-3]))

        # A shared variable for storing the iteration number.
        self.iteration = sharedX(theano._asarray(0, dtype='int32'),
                                 name='iter')

        # A shared variable for storing the annealed base learning rate, used
        # to lower the learning rate gradually after a certain amount of time.
        self.annealed = sharedX(base_lr, 'annealed')

    def learning_rate_updates(self):
        """
        Compute a dictionary of shared variable updates related to annealing
        the learning rate.

        Returns
        -------
        updates : dict
            A dictionary with the shared variables representing SGD metadata
            as keys and a symbolic expression of how they are to be updated as
            values.
        """
        ups = {}

        # Annealing coefficient. Here we're using a formula of
        # min(base_lr, anneal_start / (iteration + 1))
        if self.anneal_start is None:
            annealed = sharedX(self.base_lr)
        else:
            frac = self.anneal_start / (self.iteration + 1.)
            annealed = tensor.minimum(
                    as_floatX(frac),
                    self.base_lr  # maximum learning rate
                    )

        # Update the shared variable for the annealed learning rate.
        ups[self.annealed] = annealed
        ups[self.iteration] = self.iteration + 1

        # Calculate the learning rates for each parameter, in the order
        # they appear in self.params
        learn_rates = [annealed * self.learning_rates[p] for p in self.params]
        return ups, learn_rates

    def updates(self, gradients):
        """
        Return symbolic updates to apply given a set of gradients
        on the parameters being optimized.

        Parameters
        ----------
        gradients : list of tensor_likes
            List of symbolic gradients for the parameters contained
            in self.params, in the same order as in self.params.

        Returns
        -------
        updates : dict
            A dictionary with the shared variables in self.params as keys
            and a symbolic expression of how they are to be updated each
            SGD step as values.

        Notes
        -----
        `cost_updates` is a convenient helper function that takes all
        necessary gradients with respect to a given symbolic cost.
        """
        ups = {}
        # Add the learning rate/iteration updates
        l_ups, learn_rates = self.learning_rate_updates()
        safe_update(ups, l_ups)

        if self.use_adagrad:
            p_up = {}
            for param, gp in zip(self.params, gradients):
                acc = self.accumulators[param]
                p_up[acc] = acc + (gp ** 2).sum()
                adagrad = self.e0s[param] / (p_up[acc] ** .5)
                p_up[param] = param - adagrad * gp
        else:
            # Get the updates from sgd_updates, a PyLearn library function.
            p_up = dict(sgd_updates(self.params, gradients, learn_rates))

        # Add the things in p_up to ups
        safe_update(ups, p_up)

        # Clip the values if needed.
        # We do not want the clipping values to force an upcast
        # of the update: updates should have the same type as params
        for param, (p_min, p_max) in self.clipping_values.iteritems():
            p_min = tensor.as_tensor(p_min)
            p_max = tensor.as_tensor(p_max)
            dtype = param.dtype
            if p_min.dtype != dtype:
                p_min = tensor.cast(p_min, dtype)
            if p_max.dtype != dtype:
                p_max = tensor.cast(p_max, dtype)
            ups[param] = tensor.clip(ups[param], p_min, p_max)

        # Return the updates dictionary.
        return ups

    def cost_updates(self, cost):
        """
        Return symbolic updates to apply given a cost function.

        Parameters
        ----------
        cost : tensor_like
            Symbolic cost with respect to which the gradients of
            the parameters should be taken. Should be 0-dimensional
            (scalar valued).

        Returns
        -------
        updates : dict
            A dictionary with the shared variables in self.params as keys
            and a symbolic expression of how they are to be updated each
            SGD step as values.
        """
        grads = [tensor.grad(cost, p) for p in self.params]
        return self.updates(gradients=grads)


    def sgd_updates(self, params, grads, stepsizes):
        """Return a list of (pairs) that can be used as updates in theano.function to
        implement stochastic gradient descent.

        :param params: variables to adjust in order to minimize some cost
        :type params: a list of variables (theano.function will require shared variables)
        :param grads: the gradient on each param (with respect to some cost)
        :type grads: list of theano expressions
        :param stepsizes: step by this amount times the negative gradient on each iteration
        :type stepsizes: [symbolic] scalar or list of one [symbolic] scalar per param
        """
        try:
            iter(stepsizes)
        except Exception:
            stepsizes = [stepsizes for p in params]
        if len(params) != len(grads):
            raise ValueError('params and grads have different lens')
        updates = [(p, p - step * gp) for (step, p, gp) in zip(stepsizes, params, grads)]
        return updates

    def sgd_momentum_updates(self, params, grads, stepsizes, momentum=0.9):
        # if stepsizes is just a scalar, expand it to match params
        try:
            iter(stepsizes)
        except Exception:
            stepsizes = [stepsizes for p in params]
        try:
            iter(momentum)
        except Exception:
            momentum = [momentum for p in params]
        if len(params) != len(grads):
            raise ValueError('params and grads have different lens')
        headings = [theano.shared(numpy.zeros_like(p.get_value(borrow=True))) for p in params]
        updates = []
        for s, p, gp, m, h in zip(stepsizes, params, grads, momentum, headings):
            updates.append((p, p + s * h))
            updates.append((h, m*h - (1.0-m)*gp))
        return updates
