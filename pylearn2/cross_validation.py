"""
Cross validation module.

Each fold of cross validation is a separate experiment, so we create a
separate Train object for each model and save all of the models together.

print_monitor_average.py can be used to analyze average monitor channel
values for the collection of saved models.
"""
__author__ = "Steven Kearnes"

from copy import deepcopy
import numpy as np
import os
import warnings
try:
    from sklearn.cross_validation import (KFold, StratifiedKFold, ShuffleSplit,
                                          StratifiedShuffleSplit)
except ImportError:
    warnings.warn("Could not import from sklearn")

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.mlp import Layer, PretrainedLayer
from pylearn2.train import Train, SerializationGuard
from pylearn2.utils import safe_zip, serial


class DatasetPartitionIterator(object):
    """Constructs a new DenseDesignMatrix for each subset."""
    def __init__(self, dataset, index_iterator, return_dict=True):
        """
        Parameters
        ----------
        dataset : object
            Full dataset for use in cross validation.
        index_iterator : iterable
            Iterable that returns (train, test) or (train, valid, test) indices
            for slicing the dataset during cross validation.
        return_dict : bool
            Whether to return subset datasets as a dictionary. If True,
            returns a dict with keys 'train', 'valid', and/or 'test' (if
            index_iterator returns two slices per partition, 'train' and 'test'
            are used, and if index_iterator returns three slices per partition,
            'train', 'valid', and 'test' are used). If False, returns a list of
            datasets matching the slice order given by index_iterator.
        """
        self.dataset = dataset
        self.index_iterator = index_iterator
        dataset_iterator = dataset.iterator(mode='sequential', num_batches=1,
                                            data_specs=dataset.data_specs)
        self.dataset_iterator = dataset_iterator
        self.return_dict = return_dict

    def __iter__(self):
        for subsets in self.index_iterator:
            labels = None
            if len(subsets) == 3:
                labels = ['train', 'valid', 'test']
            elif len(subsets) == 2:
                labels = ['train', 'test']
            datasets = {}
            for i, subset in enumerate(subsets):
                subset_data = tuple(
                    fn(data[subset]) if fn else data[subset]
                    for data, fn in safe_zip(self.dataset_iterator._raw_data,
                                             self.dataset_iterator._convert))
                if len(subset_data) == 2:
                    X, y = subset_data
                else:
                    X, = subset_data
                    y = None
                dataset = DenseDesignMatrix(X=X, y=y)
                datasets[labels[i]] = dataset
            if not self.return_dict:
                datasets = tuple(datasets[label] for label in labels)
                if len(datasets) == 1:
                    datasets, = datasets
            yield datasets


class StratifiedDatasetPartitionIterator(DatasetPartitionIterator):
    """
    Subclass of DatasetPartitionIterator for stratified experiments, where
    the relative class proportions of the full dataset are maintained in
    each partition.
    """
    @staticmethod
    def get_y(dataset):
        """
        Get target values for dataset, possibly converting from one-hot
        encoding to a 1D array.

        Parameters
        ----------
        dataset : object
            Dataset containing target values for examples.
        """
        y = np.asarray(dataset.y)
        if y.ndim > 1:
            assert np.array_equal(np.unique(y), [0, 1])
            y = np.argmax(y, axis=1)
        return y


class DatasetKFold(DatasetPartitionIterator):
    """K-fold cross-validation."""
    def __init__(self, dataset, n_folds=3, indices=None, shuffle=False,
                 random_state=None):
        """
        Parameters
        ----------
        dataset : object
            Dataset to use for cross-validation.
        n_folds : int
            Number of cross-validation folds.
        indices : bool
            Whether to return indices for dataset slicing. If false, returns
            a boolean mask.
        shuffle : bool
            Whether to shuffle the dataset before partitioning.
        random_state : int or RandomState
            Random number generator used for shuffling.
        """
        n = dataset.X.shape[0]
        cv = KFold(n, n_folds, indices, shuffle, random_state)
        super(DatasetKFold, self).__init__(dataset, cv)


class StratifiedDatasetKFold(StratifiedDatasetPartitionIterator):
    """Stratified K-fold cross-validation."""
    def __init__(self, dataset, n_folds=3, indices=None):
        """
        Parameters
        ----------
        dataset : object
            Dataset to use for cross-validation.
        n_folds : int
            Number of cross-validation folds.
        indices : bool
            Whether to return indices for dataset slicing. If false, returns
            a boolean mask.
        """
        y = self.get_y(dataset)
        cv = StratifiedKFold(y, n_folds, indices)
        super(StratifiedDatasetKFold, self).__init__(dataset, cv)


class DatasetShuffleSplit(DatasetPartitionIterator):
    """Shuffle-split cross-validation."""
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None):
        """
        Parameters
        ----------
        dataset : object
            Dataset to use for cross-validation.
        n_iter : int
            Number of shuffle-split iterations.
        test_size : float, int, or None
            If float, intepreted as the proportion of examples in the test set.
            If int, interpreted as the absolute number of examples in the test
            set. If None, adjusted to the complement of train_size.
        train_size : float, int, or None
            If float, intepreted as the proportion of examples in the training
            set. If int, interpreted as the absolute number of examples in the
            training set. If None, adjusted to the complement of test_size.
        indices : bool
            Whether to return indices for dataset slicing. If false, returns
            a boolean mask.
        random_state : int or RandomState
            Random number generator used for shuffling.
        """
        n = dataset.X.shape[0]
        cv = ShuffleSplit(n, n_iter, test_size, train_size, indices,
                          random_state)
        super(DatasetShuffleSplit, self).__init__(dataset, cv)


class StratifiedDatasetShuffleSplit(StratifiedDatasetPartitionIterator):
    """Stratified shuffle-split cross-validation."""
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None):
        """
        Parameters
        ----------
        dataset : object
            Dataset to use for cross-validation.
        n_iter : int
            Number of shuffle-split iterations.
        test_size : float, int, or None
            If float, intepreted as the proportion of examples in the test set.
            If int, interpreted as the absolute number of examples in the test
            set. If None, adjusted to the complement of train_size.
        train_size : float, int, or None
            If float, intepreted as the proportion of examples in the training
            set. If int, interpreted as the absolute number of examples in the
            training set. If None, adjusted to the complement of test_size.
        indices : bool
            Whether to return indices for dataset slicing. If false, returns
            a boolean mask.
        random_state : int or RandomState
            Random number generator used for shuffling.
        """
        y = self.get_y(dataset)
        cv = StratifiedShuffleSplit(y, n_iter, test_size, train_size, indices,
                                    random_state)
        super(StratifiedDatasetShuffleSplit, self).__init__(dataset, cv)


class TrainCV(object):
    """Wrapper for Train that partitions the dataset according to CV scheme."""
    def __init__(self, dataset_iterator, model, algorithm=None,
                 save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True, save_subsets=False):
        """
        Create a Train object for each (train, valid, test) dataset with
        partitions given by the cv object.

        Parameters
        ----------
        dataset_iterator: iterable
            Cross validation iterator providing (test, train) or (test, valid,
            train) indices for partitioning the dataset.
        models: Model or array_like
            Training model. If more than one model is provided, then
        save_subsets: bool
            Whether to write individual files for each subset model.
        See docstring for Train for other argument descriptions.

        TODO: Implement checkpointing of the entire TrainCV object.
        It would be ideal to have each trainer's save() method actually write
        to a master pickle to allow easy restart. But since monitors get
        mangled when serialized, there's no way to resume training anyway.
        """
        # we need a way to save all the models without writing files
        trainers = []
        for k, datasets in enumerate(dataset_iterator):
            if save_subsets:
                path, ext = os.path.splitext(save_path)
                this_save_path = path + "-{}".format(k) + ext
                this_save_freq = save_freq
            else:
                this_save_path = None
                this_save_freq = 0

            # setup pretrained layers
            this_model = model
            if hasattr(model, 'layers') and any(
                    [isinstance(l, PretrainedLayers) for l in model.layers]):
                this_model = deepcopy(model)
                for i, layer in enumerate(this_model.layers):
                    if isinstance(layer, PretrainedLayers):
                        this_model.layers[i] = layer.select_layer(k)

            # construct an isolated Train object
            trainer = Train(datasets['train'], this_model, algorithm,
                            this_save_path, this_save_freq, extensions,
                            allow_overwrite)

            # no shared references between trainers are allowed
            trainer = deepcopy(trainer)

            trainer.algorithm._set_monitoring_dataset(datasets)
            trainers.append(trainer)
        self.trainers = trainers
        self.save_path = save_path

    def main_loop(self, time_budget=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        """
        for trainer in self.trainers:
            trainer.main_loop(time_budget)
        if self.save_path is not None:
            self.save()

    def save(self):
        """Serialize trained models."""
        try:
            models = []
            for trainer in self.trainers:
                trainer.dataset._serialization_guard = SerializationGuard()
                models.append(trainer.model)
            serial.save(self.save_path, models, on_overwrite='backup')
        finally:
            for trainer in self.trainers:
                trainer.dataset._serialization_guard = None


class PretrainedLayers(Layer):
    """Container of PretrainedLayer objects for use with TrainCV."""
    def __init__(self, layer_name, layer_content):
        """
        Parameters
        ----------
        layer_name: str
            Name of layer.
        layer_content: array_like
            Pretrained layer models for each dataset subset.
        """
        self.layer_name = layer_name
        self.layer_content = [PretrainedLayer(layer_name, subset_content)
                              for subset_content in layer_content]

    def select_layer(self, k):
        """Choose a single layer to represent. Could reassign self."""
        return self.layer_content[k]

    def set_input_space(self, space):
        return [layer.set_input_space(space) for layer in self.layer_content]

    def get_params(self):
        return self.layer_content[0].get_params()

    def get_input_space(self):
        return self.layer_content[0].get_input_space()

    def get_output_space(self):
        return self.layer_content[0].get_output_space()

    def get_monitoring_channels(self, data):
        return self.layer_content[0].get_monitoring_channels()
