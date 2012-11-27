"""
Termination criteria used to determine when to stop running a training
algorithm.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np

class TerminationCriterion(object):
    """
    A callable used to determine if a TrainingAlgorithm should quit
    running.
    """

    def __call__(self, model):
        """
        Returns True if training should continue, False otherwise
        """
        raise NotImplementedError(str(type(self))+" does not implement __call__")

class MonitorBased(TerminationCriterion):
    """
    A termination criterion that pulls out the specified channel in
    the model's monitor and checks to see if it has decreased by a
    certain proportion of the lowest value in the last N epochs.
    """
    def __init__(self, prop_decrease = .01, N = 5, channel_name=None):
        """
        Initialize a monitor-based termination criterion.

        Parameters
        ----------
        prop_decrease : float
            The threshold factor by which we expect the channel value to have
            decreased
        N : int
            Number of epochs to look back
        channel_name : string, optional
            Name of the channel to examine. If None and the monitor
            has only one channel, this channel will be used; otherwise, an
            error will be raised.
        """
        self._channel_name = channel_name
        self.prop_decrease = prop_decrease
        self.N = N
        self.countdown = N
        self.best_value = np.inf

    def __call__(self, model):
        """
        The optimization should stop if the model has run for
        N epochs without sufficient improvement.

        Parameters
        ----------
        model : Model
            The model used in the experiment and from which the monitor used
            in the termination criterion will be extracted.

        Returns
        -------
        boolean
            True if training should continue
        """
        monitor = model.monitor
        # In the case the monitor has only one channel, the channel_name can
        # be omitted and the criterion will examine the only channel
        # available. However, if the monitor has multiple channels, leaving
        # the channel_name unspecified will raise an error.
        if self._channel_name is None:
            v = monitor.channels['objective'].val_record
        else:
            v = monitor.channels[self._channel_name].val_record

        # The countdown decreases every time the termination criterion is
        # called unless the channel value is lower than the best value times
        # the prop_decrease factor, in which case the countdown is reset to N
        # and the best value is updated
        if v[- 1] < (1. - self.prop_decrease) * self.best_value:
            self.countdown = self.N
        else:
            self.countdown = self.countdown - 1

        if v[-1] < self.best_value:
            self.best_value = v[-1]

        # The optimization continues until the countdown has reached 0,
        # meaning that N epochs have passed without the model improving
        # enough.
        return self.countdown > 0

class MatchChannel(TerminationCriterion):
    """
        Stop training when a cost function reaches the same value as a cost
        function from a previous training run.
        (Useful for getting training likelihood on entire training set to
        match validation likelihood from an earlier early stopping run)
    """

    def __init__(self, channel_name, prev_channel_name, prev_monitor_name):
        self.__dict__.update(locals())
        self.target = None

    def __call__(self, model):
        if self.target is None:
            prev_monitor = getattr(model, self.prev_monitor_name)
            channels = prev_monitor.channels
            prev_channel = channels[self.prev_channel_name]
            self.target = prev_channel.val_record[-1]

        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]

        current = channel.val_record[-1]
        rval =  current > self.target
        return rval

class ChannelTarget(TerminationCriterion):
    """
        Stop training when a cost function reaches some target value.
    """

    def __init__(self, channel_name, target):
        self.__dict__.update(locals())

    def __call__(self, model):
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]

        rval =  channel.val_record[-1] > self.target
        return rval

class EpochCounter(TerminationCriterion):
    def  __init__(self, max_epochs):
        """
        A termination criterion that uses internal state to
        trigger termination after a fixed number of calls
        (epochs).

        Parameters
        ----------
        max_epochs : int
            Number of epochs (i.e. calls to this object's `__call__`
           method) after which this termination criterion should
           return `False`.
        """
        self._max_epochs = max_epochs
        self._epochs_done = 0

    def __call__(self, model):
        self._epochs_done += 1
        return self._epochs_done < self._max_epochs

class And(TerminationCriterion):
    def __init__(self, criteria):
        """
        Termination criterion representing the logical conjunction
        of several individual criteria. Optimization continues only
        if every constituent criterion returns `True`.

        Parameters
        ----------
        criteria : iterable
            A sequence of callables representing termination criteria,
            with a return value of True indicating that the gradient
            descent should continue.
        """
        self._criteria = list(criteria)

    def __call__(self, model):
        return all(criterion(model) for criterion in self._criteria)


class Or(TerminationCriterion):
    def __init__(self, criteria):
        """
        Termination criterion representing the logical disjunction
        of several individual criteria. Optimization continues if
        any of the constituent criteria return `True`.

        Parameters
        ----------
        criteria : iterable
            A sequence of callables representing termination criteria,
            with a return value of True indicating that gradient
            descent should continue.
        """
        self._criteria = list(criteria)

    def __call__(self, model):
        return any(criterion(model) for criterion in self._criteria)
