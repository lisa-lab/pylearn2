import numpy
import unittest
from nose.plugins.skip import SkipTest

from pylearn2.datasets.exc import NoDataPathError, NotInstalledError
from numpy.testing.decorators import knownfailureif

RELIES_ON_PYLEARN_ONE = True

class test_MNIST_rotated_background(unittest.TestCase):
    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def setUp(self):
        try:
            from pylearn2.datasets.icml07 import MNIST_rotated_background
            self.train = MNIST_rotated_background(which_set='train')
            self.test = MNIST_rotated_background(which_set='test')
            #doesn't define any other function
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

class test_Convex(unittest.TestCase):
    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def setUp(self):
        try:
            from pylearn2.datasets.icml07 import Convex
            self.train = Convex(which_set='train')
            self.test = Convex(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())

class test_Rectangles(unittest.TestCase):
    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def setUp(self):
        try:
            from pylearn2.datasets.icml07 import Rectangles
            self.train = Rectangles(which_set='train')
            self.test = Rectangles(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())
        
class test_RectanglesImage(unittest.TestCase):
    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def setUp(self):
        try:
            from pylearn2.datasets.icml07 import RectanglesImage
            self.train = RectanglesImage(which_set='train')
            self.test = RectangleImage(which_set = 'test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def test_get_test_set(self):
        test = self.train.get_test_set()
        numpy.testing.assert_equal(test.get_design_matrix(), self.test.get_design_matrix())
        numpy.testing.assert_equal(test.get_targets(), self.test.get_targets())

