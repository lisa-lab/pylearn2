"""module for testing datasets.stl10"""
import unittest
import numpy as np
from pylearn2.datasets import stl10
from pylearn2.space import Conv2DSpace
from pylearn2.testing.skip import skip_if_no_data


class TestSTL10(unittest.TestCase):
    """This is a unittest for stl10.py."""
    def setUp(self):
        """This loads train and test sets."""
        skip_if_no_data()
        train = stl10.STL10(which_set='train')
        test = stl10.STL10(which_set='test')

    def test_restrict(self):
        """This tests the restrict function on each fold of the train set."""
        for fold in range(10):
            train = stl10.STL10(which_set='train')
            stl10.restrict(train, fold)

#    def disabled_test_unlabeled(self):
#        """The unlabeled data is 2.4GiB.  This test is disabled by default."""
#        assert 1 == 0
#        data = stl10.STL10(which_set='unlabeled')
