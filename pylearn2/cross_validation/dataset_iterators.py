"""
Cross-validation dataset iterators.
"""
import numpy as np
import warnings

try:
    from sklearn.cross_validation import (KFold, StratifiedKFold, ShuffleSplit,
                                          StratifiedShuffleSplit)
except ImportError:
    warnings.warn("Could not import from sklearn.")

from pylearn2.cross_validation.blocks import StackedBlocksCV
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.transformer_dataset import TransformerDataset


class DatasetCV(object):
    """
    Construct a new DenseDesignMatrix for each subset.

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
    def __init__(self, dataset, index_iterator, return_dict=True):
        self.dataset = dataset
        self.index_iterator = list(index_iterator)  # allow reuse of generators
        dataset_iterator = dataset.iterator(mode='sequential', num_batches=1,
                                            data_specs=dataset.data_specs)
        self._data = tuple(dataset_iterator.next())
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
                subset_data = tuple(data[subset] for data in self._data)
                if len(subset_data) == 2:
                    X, y = subset_data
                else:
                    X, = subset_data
                    y = None
                dataset = DenseDesignMatrix(X=X, y=y)
                datasets[labels[i]] = dataset
            if not self.return_dict:
                datasets = list(datasets[label] for label in labels)
                if len(datasets) == 1:
                    datasets, = datasets
            yield datasets


class StratifiedDatasetCV(DatasetCV):
    """
    Subclass of DatasetCV for stratified experiments, where
    the relative class proportions of the full dataset are maintained in
    each partition.
    """
    @staticmethod
    def get_y(dataset):
        """
        Stratified cross-validation requires label information for
        examples. This function gets target values for a dataset,
        converting from one-hot encoding to a 1D array as needed.

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


class TransformerDatasetCV(object):
    """
    Cross-validation with dataset transformations. This class returns
    dataset subsets after transforming them with one or more pretrained
    models.

    Parameters
    ----------
    dataset_iterator : iterable
        Cross-validation iterator providing (test, train) or (test, valid,
        train) indices for partitioning the dataset.
    transformers : Model or iterable
        Transformer model(s) to use for transforming datasets.
    """
    def __init__(self, dataset_iterator, transformers):
        self.dataset_iterator = dataset_iterator
        self.transformers = transformers

    def __iter__(self):
        """
        Construct a Transformer dataset for each partition.
        """
        for k, datasets in enumerate(self.dataset_iterator):
            if isinstance(self.transformers, list):
                transformer = self.transformers[k]
            elif isinstance(self.transformers, StackedBlocksCV):
                transformer = self.transformers.select_fold(k)
            else:
                transformer = self.transformers
            if isinstance(datasets, list):
                for i, dataset in enumerate(datasets):
                    datasets[i] = TransformerDataset(dataset, transformer)
            else:
                for key, dataset in datasets.items():
                    datasets[key] = TransformerDataset(dataset, transformer)
            yield datasets


class DatasetKFold(DatasetCV):
    """
    K-fold cross-validation.

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
    def __init__(self, dataset, n_folds=3, indices=None, shuffle=False,
                 random_state=None):
        n = dataset.X.shape[0]
        cv = KFold(n, n_folds, indices, shuffle, random_state)
        super(DatasetKFold, self).__init__(dataset, cv)


class StratifiedDatasetKFold(StratifiedDatasetCV):
    """
    Stratified K-fold cross-validation.

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
    def __init__(self, dataset, n_folds=3, indices=None):
        y = self.get_y(dataset)
        cv = StratifiedKFold(y, n_folds, indices)
        super(StratifiedDatasetKFold, self).__init__(dataset, cv)


class DatasetShuffleSplit(DatasetCV):
    """
    Shuffle-split cross-validation.

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
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None):
        n = dataset.X.shape[0]
        cv = ShuffleSplit(n, n_iter, test_size, train_size, indices,
                          random_state)
        super(DatasetShuffleSplit, self).__init__(dataset, cv)


class StratifiedDatasetShuffleSplit(StratifiedDatasetCV):
    """
    Stratified shuffle-split cross-validation.

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
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None):
        y = self.get_y(dataset)
        cv = StratifiedShuffleSplit(y, n_iter, test_size, train_size, indices,
                                    random_state)
        super(StratifiedDatasetShuffleSplit, self).__init__(dataset, cv)
