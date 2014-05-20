"""
Test subset iterators.
"""
import numpy as np

from pylearn2.testing.skip import skip_if_no_sklearn


def test_validation_k_fold():
    """Test ValidationKFold."""
    skip_if_no_sklearn()
    from pylearn2.cross_validation.subset_iterators import ValidationKFold
    n = 30

    # test with indices
    cv = ValidationKFold(n)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n / cv.n_folds
        assert test.size == n / cv.n_folds


def test_stratified_validation_k_fold():
    """Test StratifiedValidationKFold."""
    skip_if_no_sklearn()
    from pylearn2.cross_validation.subset_iterators import (
        StratifiedValidationKFold)
    n = 30
    y = np.concatenate((np.zeros(n / 2, dtype=int), np.ones(n / 2, dtype=int)))

    # test with indices
    cv = StratifiedValidationKFold(y)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n / cv.n_folds
        assert test.size == n / cv.n_folds
        assert np.count_nonzero(y[valid]) == (n / 2) * (1. / cv.n_folds)
        assert np.count_nonzero(y[test]) == (n / 2) * (1. / cv.n_folds)


def test_validation_shuffle_split():
    """Test ValidationShuffleSplit."""
    skip_if_no_sklearn()
    from pylearn2.cross_validation.subset_iterators import (
        ValidationShuffleSplit)
    n = 30

    # test with indices
    cv = ValidationShuffleSplit(n)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n * cv.test_size
        assert test.size == n * cv.test_size


def test_stratified_validation_shuffle_split():
    """Test StratifiedValidationShuffleSplit."""
    skip_if_no_sklearn()
    from pylearn2.cross_validation.subset_iterators import (
        StratifiedValidationShuffleSplit)
    n = 60
    y = np.concatenate((np.zeros(n / 2, dtype=int), np.ones(n / 2, dtype=int)))

    # test with indices
    cv = StratifiedValidationShuffleSplit(y)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n
        assert valid.size == n * cv.test_size
        assert test.size == n * cv.test_size
        assert np.count_nonzero(y[valid]) == (n / 2) * cv.test_size
        assert np.count_nonzero(y[test]) == (n / 2) * cv.test_size
