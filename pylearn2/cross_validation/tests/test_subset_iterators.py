"""
Test subset iterators.
"""
import numpy as np

from pylearn2.testing.skip import skip_if_no_sklearn
from pylearn2.cross_validation.subset_iterators import (
    ValidationKFold, StratifiedValidationKFold, ValidationShuffleSplit,
    StratifiedValidationShuffleSplit)


def test_validation_k_fold():
    """Test ValidationKFold."""
    skip_if_no_sklearn()
    n = 30

    # test with indices
    cv = ValidationKFold(n, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n / cv.n_folds
        assert test.size == n / cv.n_folds

    # test with boolean masks
    cv = ValidationKFold(n, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))
        assert np.count_nonzero(valid) == n / cv.n_folds
        assert np.count_nonzero(test) == n / cv.n_folds


def test_stratified_validation_k_fold():
    """Test StratifiedValidationKFold."""
    skip_if_no_sklearn()
    n = 30
    y = np.concatenate((np.zeros(n / 2, dtype=int), np.ones(n / 2, dtype=int)))

    # test with indices
    cv = StratifiedValidationKFold(y, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n / cv.n_folds
        assert test.size == n / cv.n_folds
        assert np.count_nonzero(y[valid]) == (n / 2) * (1. / cv.n_folds)
        assert np.count_nonzero(y[test]) == (n / 2) * (1. / cv.n_folds)

    # test with boolean masks
    cv = StratifiedValidationKFold(y, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))
        assert np.count_nonzero(valid) == n / cv.n_folds
        assert np.count_nonzero(test) == n / cv.n_folds
        assert np.count_nonzero(y[valid]) == (n / 2) * (1. / cv.n_folds)
        assert np.count_nonzero(y[test]) == (n / 2) * (1. / cv.n_folds)


def test_validation_shuffle_split():
    """Test ValidationShuffleSplit."""
    skip_if_no_sklearn()
    n = 30

    # test with indices
    cv = ValidationShuffleSplit(n, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n * cv.test_size
        assert test.size == n * cv.test_size

    # test with boolean masks
    cv = ValidationShuffleSplit(n, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))
        assert np.count_nonzero(valid) == n * cv.test_size
        assert np.count_nonzero(test) == n * cv.test_size


def test_stratified_validation_shuffle_split():
    """Test StratifiedValidationShuffleSplit."""
    skip_if_no_sklearn()
    n = 60
    y = np.concatenate((np.zeros(n / 2, dtype=int), np.ones(n / 2, dtype=int)))

    # test with indices
    cv = StratifiedValidationShuffleSplit(y, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n * cv.test_size
        assert test.size == n * cv.test_size
        assert np.count_nonzero(y[valid]) == (n / 2) * cv.test_size
        assert np.count_nonzero(y[test]) == (n / 2) * cv.test_size

    # test with boolean masks
    cv = StratifiedValidationShuffleSplit(y, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))
        assert np.count_nonzero(valid) == n * cv.test_size
        assert np.count_nonzero(test) == n * cv.test_size
        assert np.count_nonzero(y[valid]) == (n / 2) * cv.test_size
        assert np.count_nonzero(y[test]) == (n / 2) * cv.test_size
