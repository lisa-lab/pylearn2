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

class MonitorBasedTermCrit(object):
    """
    A termination criterion that pulls out the specified channel in
    the model's monitor and checks to see if it has decreased by a
    certain proportion of the lowest value in the last N epochs.
    """
    def __init__(self, prop_decrease, N, channel_name=None):
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
        Returns True or False depending on whether the optimization should
        stop or not. The optimization should stop if the model has run for
        N epochs without any improvement.

        Parameters
        ----------
        model : Model
            The model used in the experiment and from which the monitor used
            in the termination criterion will be extracted.

        Returns
        -------
        boolean
            True or False, indicating if the optimization should stop or not.
        """
        monitor = model.monitor
        # In the case the monitor has only one channel, the channel_name can
        # be omitted and the criterion will examine the only channel
        # available. However, if the monitor has multiple channels, leaving
        # the channel_name unspecified will raise an error.
        if self._channel_name is None:
            v = monitor.channels['sgd_cost'].val_record
        else:
            v = monitor.channels[self._channel_name].val_record

        # The countdown decreases every time the termination criterion is
        # called unless the channel value is lower than the best value times
        # the prop_decrease factor, in which case the countdown is reset to N
        # and the best value is updated
        if v[- 1] < (1. - self.prop_decrease) * self.best_value:
            self.countdown = self.N
            self.best_value = v[-1]
        else:
            self.countdown = self.countdown - 1
        # The optimization continues until the countdown has reached 0,
        # meaning that N epochs have passed without the model improving
        # enough.
        return self.countdown > 0

