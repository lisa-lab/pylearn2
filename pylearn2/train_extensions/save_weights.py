"""
.. todo::

    WRITEME
"""
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

log = logging.getLogger(__name__)

class SaveWeights(TrainExtension):
    """
    A callback that saves a copy of the weights of the model every some given epoch frequency.
  
    Parameters
    ----------
    save_freq : int
        Frequency of saves, in epochs.
    save_path : str
        Output directory for the weights of model. 
    """
    def __init__(self, save_freq=None, save_path=None):
        assert((save_freq is not None) and (save_path is not None))
        self.save_path = save_path
        self.save_freq = save_freq
        self._tag_key = "SaveWeights"

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
        freq = self.save_freq
        epochs_seen = model.monitor.get_epochs_seen()

        if freq > 0 and epochs_seen % freq == 0:
          weights = model.get_weights()
          numpy.savez(os.path.join(self.save_path,"weights."+str(epochs_seen)+".npz"), weights)

        """
        val_record = channel.val_record
        new_cost = val_record[-1]

        if self.coeff * new_cost < self.coeff * self.best_cost:
            self.best_cost = new_cost
            # Update the tag of the model object before saving it.
            self._update_tag(model)
            if self.store_best_model:
                self.best_model = deepcopy(model)
            if self.save_path is not None:
                serial.save(self.save_path, model, on_overwrite='backup')
        """

    def _update_tag(self, model):
        """
        Update `model.tag` with information about the current best.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            The model to update.
        """
        # More stuff to be added later. For now, we care about the best cost.
        pass
        #model.tag[self._tag_key]['best_cost'] = self.best_cost
