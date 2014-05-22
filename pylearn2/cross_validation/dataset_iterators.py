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

from theano.compat import OrderedDict

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
    subset_iterator : iterable
        Iterable that returns (train, test) or (train, valid, test) indices
        or masks for partitioning the dataset during cross-validation.
    preprocessor : Preprocessor or None
        Preprocessor to apply to child datasets.
    fit_preprocessor : bool
        Whether preprocessor can fit parameters when applied to training
        data.
    which_set : str, list or None
        If None, return all subset datasets. If one or more of 'train',
        'valid', or 'test', return only the dataset(s) corresponding to the
        given subset(s).
    return_dict : bool
        Whether to return subset datasets as a dictionary. If True,
        returns a dict with keys 'train', 'valid', and/or 'test' (if
        subset_iterator returns two subsets per partition, 'train' and
        'test' are used, and if subset_iterator returns three subsets per
        partition, 'train', 'valid', and 'test' are used). If False,
        returns a list of datasets matching the subset order given by
        subset_iterator.
    """
    def __init__(self, dataset, subset_iterator, preprocessor=None,
                 fit_preprocessor=False, which_set=None, return_dict=True):
        self.dataset = dataset
        self.subset_iterator = list(subset_iterator)  # allow generator reuse
        dataset_iterator = dataset.iterator(mode='sequential', num_batches=1,
                                            data_specs=dataset.data_specs,
                                            return_tuple=True)
        self._data = dataset_iterator.next()
        self.preprocessor = preprocessor
        self.fit_preprocessor = fit_preprocessor
        self.which_set = which_set
        if which_set is not None:
            which_set = np.atleast_1d(which_set)
            assert len(which_set)
            for label in which_set:
                if label not in ['train', 'valid', 'test']:
                    raise ValueError("Unrecognized subset '{}'".format(label))
            self.which_set = which_set
        self.return_dict = return_dict

    def get_data_subsets(self):
        """
        Partition the dataset according to cross-validation subsets and
        return the raw data in each subset.
        """
        for subsets in self.subset_iterator:
            labels = None
            if len(subsets) == 3:
                labels = ['train', 'valid', 'test']
            elif len(subsets) == 2:
                labels = ['train', 'test']
            # data_subsets is an OrderedDict to maintain label order
            data_subsets = OrderedDict()
            for i, subset in enumerate(subsets):
                subset_data = tuple(data[subset] for data in self._data)
                if len(subset_data) == 2:
                    X, y = subset_data
                else:
                    X, = subset_data
                    y = None
                data_subsets[labels[i]] = (X, y)
            yield data_subsets

    def __iter__(self):
        """
        Create a DenseDesignMatrix for each dataset subset and apply any
        preprocessing to the child datasets.
        """
        for data_subsets in self.get_data_subsets():
            datasets = {}
            for label, data in data_subsets.items():
                X, y = data
                datasets[label] = DenseDesignMatrix(X=X, y=y)

            # preprocessing
            if self.preprocessor is not None:
                self.preprocessor.apply(datasets['train'],
                                        can_fit=self.fit_preprocessor)
                for label, dataset in datasets.items():
                    if label == 'train':
                        continue
                    self.preprocessor.apply(dataset, can_fit=False)

            # which_set
            if self.which_set is not None:
                for label, dataset in datasets.items():
                    if label not in self.which_set:
                        del datasets[label]
                        del data_subsets[label]
                if not len(datasets):
                    raise ValueError("No matching dataset(s) for " +
                                     "{}".format(self.which_set))

            if not self.return_dict:
                # data_subsets is an OrderedDict to maintain label order
                datasets = list(datasets[label]
                                for label in data_subsets.keys())
                if len(datasets) == 1:
                    datasets, = datasets
            yield datasets


class StratifiedDatasetCV(DatasetCV):
    """
    Subclass of DatasetCV for stratified experiments, where
    the relative class proportions of the full dataset are maintained in
    each partition.

    Parameters
    ----------
    dataset : object
        Full dataset for use in cross validation.
    subset_iterator : iterable
        Iterable that returns (train, test) or (train, valid, test) indices
        or masks for partitioning the dataset during cross-validation.
    preprocessor : Preprocessor or None
        Preprocessor to apply to child datasets.
    fit_preprocessor : bool
        Whether preprocessor can fit parameters when applied to training
        data.
    which_set : str, list or None
        If None, return all subset datasets. If one or more of 'train',
        'valid', or 'test', return only the dataset(s) corresponding to the
        given subset(s).
    return_dict : bool
        Whether to return subset datasets as a dictionary. If True,
        returns a dict with keys 'train', 'valid', and/or 'test' (if
        subset_iterator returns two subsets per partition, 'train' and
        'test' are used, and if subset_iterator returns three subsets per
        partition, 'train', 'valid', and 'test' are used). If False,
        returns a list of datasets matching the subset order given by
        subset_iterator.
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
    dataset_iterator : DatasetCV
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
    kwargs : dict
        Keyword arguments for DatasetCV.
    """
    def __init__(self, dataset, n_folds=3, indices=True, shuffle=False,
                 random_state=None, **kwargs):
        n = dataset.X.shape[0]
        cv = KFold(n, n_folds, indices, shuffle, random_state)
        super(DatasetKFold, self).__init__(dataset, cv, **kwargs)


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
    kwargs : dict
        Keyword arguments for DatasetCV.
    """
    def __init__(self, dataset, n_folds=3, indices=True, **kwargs):
        y = self.get_y(dataset)
        cv = StratifiedKFold(y, n_folds, indices)
        super(StratifiedDatasetKFold, self).__init__(dataset, cv, **kwargs)


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
    kwargs : dict
        Keyword arguments for DatasetCV.
    """
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None, **kwargs):
        n = dataset.X.shape[0]
        cv = ShuffleSplit(n, n_iter, test_size, train_size, indices,
                          random_state)
        super(DatasetShuffleSplit, self).__init__(dataset, cv, **kwargs)


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
    kwargs : dict
        Keyword arguments for DatasetCV.
    """
    def __init__(self, dataset, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None, **kwargs):
        y = self.get_y(dataset)
        cv = StratifiedShuffleSplit(y, n_iter, test_size, train_size, indices,
                                    random_state)
        super(StratifiedDatasetShuffleSplit, self).__init__(dataset, cv,
                                                            **kwargs)
