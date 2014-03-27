"""
Cross validation module.

Each fold of cross validation is a separate experiment, so we create a separate
Train object for each model and save all of the models together, as well as the
cross validation object for future use.
"""
__author__ = "Steven Kearnes"

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import safe_zip
from sklearn.cross_validation import *


class DatasetIterator(object):
    """Returns a new DenseDesignMatrix for each subset."""
    def __init__(self, dataset, index_iterator, return_dict=True):
        self.index_iterator = index_iterator
        targets = False
        if dataset.get_targets() is not None:
            targets = True
        dataset_iterator = dataset.iterator(mode='sequential', num_batches=1,
                                            targets=targets)
        self.dataset_iterator = dataset_iterator
        self.return_dict = return_dict

    def __iter__(self):
        for subsets in self.index_iterator:
            labels = []
            if len(subsets) == 1:
                labels = ['train']
            elif len(subsets) == 2:
                labels = ['train', 'test']
            elif len(subsets) == 3:
                labels = ['train', 'valid', 'test']
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


class DatasetKFold(DatasetIterator):
    def __init__(self, dataset, n_folds=3, indices=None, shuffle=False,
                 random_state=None):
        n = dataset.X.shape[0]
        cv = KFold(n, n_folds, indices, shuffle, random_state)
        super(DatasetKFold, self).__init__(dataset, cv)


class DatasetStratifiedKFold(StratifiedKFold):
    def __init__(self, dataset, n_folds=3, indices=None):
        y = dataset.y
        super(DatasetStratifiedKFold, self).__init__(y, n_folds, indices)
        self.dataset = dataset


class TrainCV(object):
    """Wrapper for Train that partitions the dataset according to CV scheme."""
    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 save_freq=0, extensions=None, allow_overwrite=True):
        """
        Create a Train object for each (train, valid, test) dataset with
        partitions given by the cv object.

        Parameters
        ----------
        cv: iterable
            Cross validation iterator providing (test, train) or (test, valid,
            train) indices for partitioning the dataset.
        See docstring for Train for other argument descriptions.
        """
        # need to figure out how to get a slice from a database
