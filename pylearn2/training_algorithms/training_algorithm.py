"""Module defining the interface for training algorithms."""
from pylearn2.datasets.dataset import Dataset

class TrainingAlgorithm(object):
    """
    An abstract superclass that defines the interface of training
    algorithms.
    """

    def _register_update_callbacks(self, update_callbacks):
        """
        .. todo::

            WRITEME
        """
        if update_callbacks is None:
            update_callbacks = []
        # If it's iterable, we're fine. If not, it's a single callback,
        # so wrap it in a list.
        try:
            iter(update_callbacks)
            self.update_callbacks = update_callbacks
        except TypeError:
            self.update_callbacks = [update_callbacks]

    def setup(self, model, dataset):
        """
        Initialize the given training algorithm.

        Parameters
        ----------
        model : object
            Object that implements the Model interface defined in
            `pylearn2.models`.
        dataset : object
            Object that implements the Dataset interface defined in
            `pylearn2.datasets`.

        Notes
        -----
        Called by the training script prior to any calls involving data.
        This is a good place to compile theano functions for doing learning.
        """
        self.model = model

    def train(self, dataset):
        """
        Performs some amount of training, generally one "epoch" of online
        learning

        Parameters
        ----------
        dataset : object
            Object implementing the dataset interface defined in
            `pylearn2.datasets.dataset.Dataset`.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _set_monitoring_dataset(self, monitoring_dataset):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        monitoring_dataset : None or Dataset or dict
            None for no monitoring, or Dataset, to monitor on one dataset,
            or dict mapping string names to Datasets
        """
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }
        else:
            if monitoring_dataset is not None:
                assert isinstance(monitoring_dataset, dict)
                for key in monitoring_dataset:
                    assert isinstance(key, str)
                    value = monitoring_dataset[key]
                    if not isinstance(value, Dataset):
                        raise TypeError("Monitoring dataset with name " + key +
                                        " is not a dataset, it is a " +
                                        str(type(value)))
            self.monitoring_dataset = monitoring_dataset

    def continue_learning(self, model):
        """
        Return True to continue learning. Called after the Monitor
        has been run on the latest parameters so the monitor may be used
        to determine convergence.

        Parameters
        ----------
        model : WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "continue_learning.")

    def _synchronize_batch_size(self, model):
        """
        Adapts `self.batch_size` to be consistent with `model`

        Parameters
        ----------
        model : Model
            The model to synchronize the batch size with
        """
        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size and model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError("batch_size argument to " +
                                             str(type(self)) +
                                             "conflicts with model's " +
                                             "force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        if self.batch_size is None:
            raise NoBatchSizeError()

class NoBatchSizeError(ValueError):
    """
    An exception raised when the user does not specify a batch size anywhere.
    """
    def __init__(self):
        super(NoBatchSizeError, self).__init__("Neither the "
                "TrainingAlgorithm nor the model were given a specification "
                "of the batch size.")

