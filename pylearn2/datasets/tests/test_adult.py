"""
Test code for adult.py
"""
import numpy
from pylearn2.datasets.adult import adult
from pylearn2.testing.skip import skip_if_no_data


def test_adult():
    """
    Tests if it will work correctly for train and test set.
    """
    skip_if_no_data()
    adult_train = adult(which_set='train')
    assert (adult_train.X >= 0.).all()
    assert adult_train.y.dtype == bool
    assert adult_train.X.shape == (30162, 104)
    assert adult_train.y.shape == (30162, 1)

    adult_test = adult(which_set='test')
    assert (adult_test.X >= 0.).all()
    assert adult_test.y.dtype == bool
    assert adult_test.X.shape == (15060, 103)
    assert adult_test.y.shape == (15060, 1)
