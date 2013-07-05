import numpy
import unittest
from pylearn2.datasets.icml07 import MNIST_rotated_background, Convex, Rectangles, RectanglesImage
from pylearn2.testing.skip import skip_if_no_data

class test_MNIST_rotated_background(unittest.TestCase):
    def setUp(self):
        #WARN: relies on pylearn1 to load icml07 data. Should not work.
        skip_if_no_data()
        self.train = MNIST_rotated_background(which_set='train')
        self.test = MNIST_rotated_background(which_set='test')
        #doesn't define any other function

class test_Convex(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = Convex(which_set='train')
        self.test = Convex(which_set='test')

    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())

class test_Rectangles(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = Rectangles(which_set='train')
        self.test = Rectangles(which_set='test')

    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())
        
class test_RectanglesImage(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = RectanglesImage(which_set='train')
        self.test = RectangleImage(which_set = 'test')

    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())

