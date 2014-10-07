"""module for testing datasets.stl10"""
import unittest
from pylearn2.datasets import stl10
from pylearn2.testing.skip import skip_if_no_data


class TestSTL10(unittest.TestCase):

    """
    This is a unittest for stl10.py.

    Parameters
    ----------
    none

    """

    def setUp(self):
        """This loads train and test sets."""
        skip_if_no_data()
        data = stl10.STL10(which_set='train')
        data = stl10.STL10(which_set='test')

    def test_restrict(self):
        """This tests the restrict function on each fold of the train set."""
        for fold in range(10):
            train = stl10.STL10(which_set='train')
            stl10.restrict(train, fold)

#    def disabled_test_unlabeled(self):
#        """The unlabeled data is 2.4GiB.  This test is disabled by default."""
#        data = stl10.STL10(which_set='unlabeled')
