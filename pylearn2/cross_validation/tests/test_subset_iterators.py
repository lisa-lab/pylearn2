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
    n = 100

    # test with indices
    cv = ValidationKFold(n, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n

    # test with boolean masks
    cv = ValidationKFold(n, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))


def test_stratified_validation_k_fold():
    """Test StratifiedValidationKFold."""
    skip_if_no_sklearn()
    n = 100
    y = np.random.randint(2, size=n)

    # test with indices
    cv = StratifiedValidationKFold(y, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n

    # test with boolean masks
    cv = StratifiedValidationKFold(y, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))


def test_validation_shuffle_split():
    """Test ValidationShuffleSplit."""
    skip_if_no_sklearn()
    n = 100

    # test with indices
    cv = ValidationShuffleSplit(n, indices=True)
    for train, valid, test in cv:
        assert np.unique(np.concatenate((train, valid, test))).size == n

    # test with boolean masks
    cv = ValidationShuffleSplit(n, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))


def test_stratified_validation_shuffle_split():
    """Test StratifiedValidationShuffleSplit."""
    skip_if_no_sklearn()
    n = 100
    y = np.random.randint(2, size=n)

    # test with indices
    cv = StratifiedValidationShuffleSplit(y, indices=True)
    try:
        for train, valid, test in cv:
            assert np.unique(np.concatenate((train, valid, test))).size == n
    except AssertionError:
        import IPython
        IPython.embed()

    # test with boolean masks
    cv = StratifiedValidationShuffleSplit(y, indices=False)
    for train, valid, test in cv:
        assert not np.any(np.logical_and(train, valid))
        assert not np.any(np.logical_and(train, test))
        assert not np.any(np.logical_and(valid, test))
        assert np.all(np.logical_or(np.logical_or(train, valid), test))
