import numpy
import unittest
from pylearn2.datasets.stl10 import STL10, restrict
from pylearn2.testing.skip import skip_if_no_data

class TestSTL10(unittest.TestCase):
    def setUp(self):
        skip_if_no_data('stl10')
        skip_if_no_data('stl10_matlab')
        self.train = STL10(which_set='train')
        self.test = STL10(which_set='test') 

    def test_restrict(self):
        #restrict provides a series of tests that are comprehensive enough
        #right now.
        restrict(self.train, 0)
        restrict(self.test, 0)
