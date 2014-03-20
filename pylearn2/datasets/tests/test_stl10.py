import unittest
import numpy as np
from pylearn2.datasets.stl10 import STL10
from pylearn2.space import Conv2DSpace
from pylearn2.testing.skip import skip_if_no_data


class TestSTL10(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = STL10(which_set='train')
        self.test = STL10(which_set='test')
        self.unlabeled = STL10(which_set='unlabeled')

    def test_restrict(self):
        for fold in range(10):
            restrict(self.train, fold)
