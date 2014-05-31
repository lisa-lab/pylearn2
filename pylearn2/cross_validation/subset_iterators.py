"""
Cross-validation subset iterators.

The cross-validation iterators in sklearn only return train/test splits.
Several of the subset iterators in this module return train/valid/test
splits by starting with a train/test split and further dividing the train
subset into a train/valid split.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import warnings
try:
    from sklearn.cross_validation import (KFold, StratifiedKFold, ShuffleSplit,
                                          StratifiedShuffleSplit)
except ImportError:
    warnings.warn("Could not import from sklearn.")
    KFold = StratifiedKFold = ShuffleSplit = StratifiedShuffleSplit = object


def get_k_fold_splits(cv):
    """
    Use the test set from the next fold as the validation set for this
    fold. Since the test sets are chosen sequentially by KFold, the test
    set for the next fold will never contain any indices from the current
    test set.

    Parameters
    ----------
    cv : list
        Cross-validation splits.
    """
    for k, (train, test) in enumerate(cv):
        valid_k = k + 1
        if valid_k == len(cv):
            valid_k = 0
        valid = cv[valid_k][1]
        train = np.setdiff1d(train, valid)
        yield train, valid, test


def get_validation_set_from_train(train, train_cv):
    """
    Repartition training set into training and validation sets using the
    given subset iterator. Only the first train/test split of train_cv is
    used.

    Parameters
    ----------
    train : array_like
        Indices corresponding to the training set.
    train_cv : subset iterator
        Cross-validation iterator that returns train/test splits of the
        training set.
    """
    for new_train, new_valid in train_cv:
        return train[new_train], train[new_valid]


class ValidationKFold(KFold):
    """
    K-fold cross-validation. One fold is used for testing, another for
    validation, and the remainder for training.

    Parameters
    ----------
    n : int
        Number of examples.
    n_folds : int
        Number of cross-validation folds. Must be at least 3.
    shuffle : bool
        Whether to shuffle the data before splitting.
    random_state : int, RandomState, or None
        Pseudorandom number seed or generator to use for shuffling.
    """
    def __init__(self, n, n_folds=3, shuffle=False, random_state=None):
        if n_folds <= 2:
            raise ValueError("k-fold cross-validation requires at least one " +
                             "train / valid / test split by setting " +
                             "n_folds=3 or more, got " +
                             "n_folds={}.".format(n_folds))
        super(ValidationKFold, self).__init__(n, n_folds=n_folds,
                                              shuffle=shuffle,
                                              random_state=random_state)

    def __iter__(self):
        """Yield train/valid/test splits."""
        cv = list(super(ValidationKFold, self).__iter__())
        for train, valid, test in get_k_fold_splits(cv):
            yield train, valid, test


class StratifiedValidationKFold(StratifiedKFold):
    """
    Stratified K-fold cross-validation. One fold is used for testing,
    another for validation, and the remainder for training.

    Parameters
    ----------
    y : array_like
        Labels for examples.
    n_folds : int
        Number of cross-validation folds. Must be at least 3.
    shuffle : bool
        Whether to shuffle the data before splitting.
    random_state : int, RandomState, or None
        Pseudorandom number seed or generator to use for shuffling.
    """
    def __init__(self, y, n_folds=3, shuffle=False, random_state=None):
        if n_folds <= 2:
            raise ValueError("k-fold cross-validation requires at least one " +
                             "train/valid/test split by setting n_folds=3 " +
                             "or more, got n_folds={}.".format(n_folds))
        try:
            super(StratifiedValidationKFold, self).__init__(
                y, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
        except TypeError:
            assert not shuffle and not random_state, (
                "The 'shuffle' and 'random_state' arguments are not " +
                "supported by this version of sklearn. See "
                "http://scikit-learn.org/stable/developers/index.html" +
                "#git-repo for details on installing the development version.")
            super(StratifiedValidationKFold, self).__init__(y, n_folds=n_folds)

    def __iter__(self):
        """Yield train/valid/test splits."""
        cv = list(super(StratifiedValidationKFold, self).__iter__())
        for train, valid, test in get_k_fold_splits(cv):
            yield train, valid, test


class ValidationShuffleSplit(ShuffleSplit):
    """
    Random permutation cross-validation. The training set is further split
    into training and validation subsets.

    Note that n_train, n_test, etc. are not updated to reflect the new
    divisions.

    Parameters
    ----------
    n : int
        Number of examples.
    n_iter : int
        Number of shuffle/split iterations.
    test_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to the complement
        of train_size + valid_size.
    valid_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to match
        test_size.
    train_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to the complement
        of valid_size + test_size.
    random_state : int, RandomState, or None
        Pseudorandom number seed or generator to use for shuffling.
    """
    def __init__(self, n, n_iter=10, test_size=0.1, valid_size=None,
                 train_size=None, random_state=None):
        super(ValidationShuffleSplit, self).__init__(n, n_iter=n_iter,
                                                     test_size=test_size,
                                                     train_size=train_size,
                                                     random_state=random_state)
        if valid_size is None:
            valid_size = self.n_test

        # correct proportion to correspond to a subset of the training set
        if valid_size < 1.0:
            valid_size /= 1.0 - np.true_divide(self.n_test, self.n)
        self.valid_size = valid_size

    def __iter__(self):
        """
        Return train/valid/test splits. The validation set is generated by
        splitting the training set.
        """
        for train, test in super(ValidationShuffleSplit, self).__iter__():
            n = len(np.arange(self.n)[train])
            train_cv = ShuffleSplit(n, test_size=self.valid_size,
                                    random_state=self.random_state)
            train, valid = get_validation_set_from_train(train, train_cv)
            yield train, valid, test


class StratifiedValidationShuffleSplit(StratifiedShuffleSplit):
    """
    Random stratified permutation cross-validation. The training set is
    further split into training and validation subsets.

    Note that n_train, n_test, etc. are not updated to reflect the new
    divisions.

    Parameters
    ----------
    y : array_like
        Labels for examples.
    n_iter : int
        Number of shuffle/split iterations.
    test_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to the complement
        of train_size + valid_size.
    valid_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to match
        test_size.
    train_size : float, int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the entire dataset to include in the validation
        split. If int, represents the absolute number of validation
        samples. If None, the value is automatically set to the complement
        of valid_size + test_size.
    random_state : int, RandomState, or None
        Pseudorandom number seed or generator to use for shuffling.
    """
    def __init__(self, y, n_iter=10, test_size=0.1, valid_size=None,
                 train_size=None, random_state=None):
        super(StratifiedValidationShuffleSplit, self).__init__(
            y, n_iter=n_iter, test_size=test_size, train_size=train_size,
            random_state=random_state)
        if valid_size is None:
            valid_size = self.n_test

        # correct proportion to correspond to a subset of the training set
        if valid_size < 1.0:
            valid_size /= 1.0 - np.true_divide(self.n_test, self.n)
        self.valid_size = valid_size

    def __iter__(self):
        """
        Return train/valid/test splits. The validation set is generated by
        a stratified split of the training set.
        """
        for train, test in super(
                StratifiedValidationShuffleSplit, self).__iter__():
            y = self.y[train]
            train_cv = StratifiedShuffleSplit(y, test_size=self.valid_size,
                                              random_state=self.random_state)
            train, valid = get_validation_set_from_train(train, train_cv)
            yield train, valid, test
