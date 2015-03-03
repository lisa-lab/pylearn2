"""TrainExtensions for keeping track of and saving the best
   parameters during training. TODO: fill out properly."""
__authors__ = "XXX"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["XXX", "YYY"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from copy import deepcopy
import logging
import os.path
import socket
import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.utils.timing import log_timing


log = logging.getLogger(__name__)


class KeepBestParams(TrainExtension):
    """
    A callback which keeps track of a model's best parameters based on its
    performance for a given cost on a given dataset.

    Parameters
    ----------
    model : pylearn2.models.model.Model
        the model whose best parameters we want to keep track of
    cost : tensor_like
        cost function used to evaluate the model's performance
    monitoring_dataset : pylearn2.datasets.dataset.Dataset
        dataset on which to compute the cost
    batch_size : int
        size of the batches used to compute the cost
    """

    def __init__(self, model, cost, monitoring_dataset, batch_size):
        self.model = model
        self.cost = cost
        self.dataset = monitoring_dataset
        self.batch_size = batch_size
        self.minibatch = T.matrix('minibatch')
        self.target = T.matrix('target')
        if cost.supervised:
            self.supervised = True
            self.cost_function = theano.function(inputs=[self.minibatch,
                                                         self.target],
                                                 outputs=cost(model,
                                                              self.minibatch,
                                                              self.target))
        else:
            self.supervised = False
            self.cost_function = theano.function(inputs=[self.minibatch],
                                                 outputs=cost(model,
                                                              self.minibatch))
        self.best_cost = numpy.inf
        self.best_params = model.get_param_values()

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, records the model's parameters.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            Not used
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        if self.supervised:
            it = self.dataset.iterator('sequential',
                                       batch_size=self.batch_size,
                                       targets=True)
            new_cost = numpy.mean([self.cost_function(minibatch, target)
                                   for minibatch, target in it])
        else:
            it = self.dataset.iterator('sequential',
                                       batch_size=self.batch_size,
                                       targets=False)
            new_cost = numpy.mean([self.cost_function(minibatch)
                                   for minibatch in it])
        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_params = self.model.get_param_values()

    def get_best_params(self):
        """Returns the best parameters up to now for the model."""
        return self.best_params


class MonitorBasedSaveBest(TrainExtension):
    """
    A callback that saves a copy of the model every time it achieves a new
    minimal value of a monitoring channel. Also stores the best model in
    memory.

    Parameters
    ----------
    channel_name : str
        The name of the monitor channel we want to minimize.
    save_path : str or None, optional
        Output filename for best model. If None (the default),
        store_best_model must be True.
    store_best_model : bool, optional
        Whether to store the best model in memory. If False (the default),
        save_path must be defined.
    start_epoch : int, optional
        After the specified epoch, the model will start to be saved. Setting
        this value to a reasonable value prevents the library from saving the
        model too many times at the beginning of training.
    higher_is_better : bool, optional
        Whether a higher value of channel_name indicates a better model.
    tag_key : str, optional
        A unique key to use for storing diagnostic information in
        `model.tag`. If `None`, use the class name (default).
    """
    def __init__(self, channel_name, save_path=None, store_best_model=False,
                 start_epoch=0, higher_is_better=False, tag_key=None):
        self.channel_name = channel_name
        assert save_path is not None or store_best_model, (
            "Either save_path must be defined or store_best_model must be " +
            "True. (Or both.)")
        self.save_path = save_path
        self.store_best_model = store_best_model
        self.start_epoch = start_epoch
        self.higher_is_better = higher_is_better
        if higher_is_better:
            self.coeff = -1.
        else:
            self.coeff = 1.

        # If no tag key is provided, use the class name by default.
        if tag_key is None:
            tag_key = self.__class__.__name__
        self._tag_key = tag_key

        # placeholders
        self.best_cost = self.coeff * np.inf
        self.best_model = None

    def setup(self, model, dataset, algorithm):
        """
        Sets some model tag entries.

        Parameters
        ----------
        model : pylearn2.models.model.Model
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        if self._tag_key in model.tag:
            log.warning('Model tag key "%s" already found. This may indicate '
                        'multiple instances of %s trying to use the same tag '
                        'entry.',
                        self._tag_key, self.__class__.__name__)
            log.warning('If this is the case, specify tag key manually in '
                        '%s constructor.', self.__class__.__name__)
        # This only needs to be written once.
        model.tag[self._tag_key]['channel_name'] = self.channel_name
        # Useful information for locating the saved model.
        if self.save_path is not None:
            model.tag[self._tag_key]['save_path'] = os.path.abspath(
                self.save_path)
        model.tag[self._tag_key]['hostname'] = socket.gethostname()
        self._update_tag(model)

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            model.monitor must contain a channel with name given by
            self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if self.coeff * new_cost < self.coeff * self.best_cost and \
           monitor._epochs_seen >= self.start_epoch:
            self.best_cost = new_cost
            # Update the tag of the model object before saving it.
            self._update_tag(model)
            if self.store_best_model:
                self.best_model = deepcopy(model)
            if self.save_path is not None:
                with log_timing(log, 'Saving to ' + self.save_path):
                    serial.save(self.save_path, model, on_overwrite='backup')

    def _update_tag(self, model):
        """
        Update `model.tag` with information about the current best.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            The model to update.
        """
        # More stuff to be added later. For now, we care about the best cost.
        model.tag[self._tag_key]['best_cost'] = self.best_cost
