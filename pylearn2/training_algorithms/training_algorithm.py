from pylearn2.datasets.dataset import Dataset

class TrainingAlgorithm(object):
    """
    An abstract superclass that defines the interface of training
    algorithms.
    """
    def _register_update_callbacks(self, update_callbacks):
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
        status : bool
            `True` if the algorithm wishes to continue for another epoch.
            `False` if the algorithm has converged.
        """
        raise NotImplementedError()

    def _set_monitoring_dataset(self, monitoring_dataset):
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }
        else:
            for key in monitoring_dataset:
                assert isinstance(key, str)
                assert isinstance(monitoring_dataset[key], Dataset)
            self.monitoring_dataset = monitoring_dataset
