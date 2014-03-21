import unittest
import numpy as np
from pylearn2.datasets import stl10
from pylearn2.space import Conv2DSpace
from pylearn2.testing.skip import skip_if_no_data


class TestSTL10(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = stl10.STL10(which_set='train')
        self.test = stl10.STL10(which_set='test')

    def test_restrict(self):
        for fold in range(10):
            train = stl10.STL10(which_set='train')
            stl10.restrict(train, fold)

    def disabled_test_unlabeled(self):
        # this dataset is 2.4GiB
        data = stl10.STL10(which_set='unlabeled')
