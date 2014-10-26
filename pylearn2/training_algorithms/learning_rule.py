"""
A module containing different learning rules for use with the SGD training
algorithm.
"""
import numpy as np
import warnings

from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import sharedX
from pylearn2.utils import wraps


class LearningRule():
    """
    A pylearn2 learning rule is an object which computes new parameter values
    given (1) a learning rate (2) current parameter values and (3) the current
    estimated gradient.
    """

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Method called by the training algorithm, which allows LearningRules to
        add monitoring channels.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        raise NotImplementedError()

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        e.g. for standard SGD, one would return `sgd_rule_updates` defined
        below. Note that such a `LearningRule` object is not implemented, as
        these updates are implemented by default when the `learning_rule`
        parameter of sgd.SGD.__init__ is None.

        .. code-block::  python

            sgd_rule_updates = OrderedDict()
            for (param, grad) in grads.iteritems():
                sgd_rule_updates[k] = (param - learning_rate *
                                       lr_scalers.get(param, 1.) * grad)
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_updates.")


class Momentum(LearningRule):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.

    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc

    Parameters
    ----------
    init_momentum : float
        Initial value for the momentum coefficient. It remains fixed during
        training unless used with a `training_algorithms.sgd.MomentumAdjustor`
        extension.
    nesterov_momentum: bool
        Use the accelerated momentum technique described in:
        "Advances in Optimizing Recurrent Networks", Yoshua Bengio, et al.

    """

    def __init__(self, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')
        self.nesterov_momentum = nesterov_momentum

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Activates monitoring of the momentum.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        monitor.add_channel(
            name='momentum',
            ipt=None,
            val=self.momentum,
            data_specs=(NullSpace(), ''),
            dataset=monitoring_dataset)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the updates for learning with gradient descent + momentum.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """

        updates = OrderedDict()

        for (param, grad) in grads.iteritems():
            vel = sharedX(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert grad.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = learning_rate * lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * grad

            inc = updates[vel]
            if self.nesterov_momentum:
                inc = self.momentum * inc - scaled_lr * grad

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        return updates


class MomentumAdjustor(TrainExtension):
    """
    A TrainExtension that implements a linear momentum schedule.

    Parameters
    ----------
    final_momentum : float
        The momentum coefficient to use at the end of learning.
    start : int
        The epoch on which to start growing the momentum coefficient.
    saturate : int
        The epoch on which the moment should reach its final value.
    """
    def __init__(self, final_momentum, start, saturate):
        if saturate < start:
            raise TypeError("Momentum can't saturate at its maximum value " +
                            "before it starts increasing.")

        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0

    def on_monitor(self, model, dataset, algorithm):
        """
        Updates the momentum according to the linear schedule.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model to which the training algorithm is applied.
        dataset : pylearn2.datasets.Dataset
            The dataset to which the model is applied.
        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            Describes how gradients should be updated.
        """
        if hasattr(algorithm, 'learning_rule'):
            momentum = algorithm.learning_rule.momentum
        else:
            # TODO: remove once training_algorithm.sgd.SGD(init_momentum)
            # is officially deprecated.
            momentum = algorithm.momentum

        if not self._initialized:
            self._init_momentum = momentum.get_value()
            self._initialized = True
        self._count += 1
        momentum.set_value(np.cast[config.floatX](self.current_momentum()))

    def current_momentum(self):
        """Returns the momentum currently desired by the schedule."""
        w = self.saturate - self.start

        if w == 0:
            # saturate=start, so just jump straight to final momentum
            if self._count >= self.start:
                return self.final_momentum
            return self._init_momentum

        alpha = float(self._count - self.start) / float(w)
        if alpha < 0.:
            alpha = 0.
        if alpha > 1.:
            alpha = 1.
        return self._init_momentum * (1 - alpha) + alpha * self.final_momentum


class AdaDelta(LearningRule):
    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.

    Parameters
    ----------
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper.
    """

    def __init__(self, decay=0.95):
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        # TODO: add channels worth monitoring
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
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
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(grads[param])
            )

            # Compute update
            epsilon = lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

        return updates


class RMSProp(LearningRule):
    """
    Implements the RMSProp learning rule.

    The RMSProp learning rule is described by Hinton in `lecture 6
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`
    of the Coursera Neural Networks for Machine Learning course.

    In short, Hinton suggests "[the] magnitude of the gradient can be very
    different for different weights and can change during learning.  This
    makes it hard to choose a global learning rate." RMSProp solves this
    problem by "[dividing] the learning rate for a weight by a running
    average of the magnitudes of recent gradients for that weight."


    Parameters
    ----------
    decay : float, optional
        Decay constant similar to that used in AdaDelta and Momentum methods.
    max_scaling: float, optional
        Restrict the RMSProp gradient scaling coefficient to values
        below `max_scaling`.

    Notes
    -----
    An instance of this LearningRule should only be used with one
    TrainingAlgorithm, and its get_updates method should be called
    only once. This is required in order to make the monitoring
    channels correctly report the moving averages.
    """

    def __init__(self, decay=0.9, max_scaling=1e5):
        assert 0. <= decay < 1.
        assert max_scaling > 0
        self.decay = sharedX(decay, 'decay')
        self.epsilon = 1. / max_scaling
        self.mean_square_grads = OrderedDict()

    @wraps(LearningRule.add_channels_to_monitor)
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        The channels added are the min, mean, and max of the
        mean_square_grad of each parameter.
        """

        channel_mapping = {
            '_min': T.min,
            '_max': T.max,
            '_mean': T.mean
        }

        for mean_square_grad in self.mean_square_grads.values():
            for suffix, op in channel_mapping.items():
                monitor.add_channel(
                    name=(mean_square_grad.name + suffix),
                    ipt=None,
                    val=op(mean_square_grad),
                    data_specs=(NullSpace(), ''),
                    dataset=monitoring_dataset)
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule. See Notes for side-effects.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        This method has the side effect of storing the moving average
        of the square gradient in `self.mean_square_grads`. This is
        necessary in order for the monitoring channels to be able
        to track the value of these moving averages.
        Therefore, this method should only get called once for each
        instance of RMSProp.
        """

        updates = OrderedDict()
        for param in grads:

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)

            if param.name is None:
                raise ValueError("Model parameters must be named.")
            mean_square_grad.name = 'mean_square_grad_' + param.name

            if param.name in self.mean_square_grads:
                warnings.warn("Calling get_updates more than once on the "
                              "gradients of `%s` may make monitored values "
                              "incorrect." % param.name)
            # Store variable in self.mean_square_grads for monitoring.
            self.mean_square_grads[param.name] = mean_square_grad

            # Accumulate gradient
            new_mean_squared_grad = (self.decay * mean_square_grad +
                                     (1 - self.decay) * T.sqr(grads[param]))

            # Compute update
            scaled_lr = lr_scalers.get(param, 1.) * learning_rate
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            delta_x_t = - scaled_lr * grads[param] / rms_grad_t

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[param] = param + delta_x_t

        return updates
