import numpy
import unittest
from pylearn2.datasets.svhn import SVHN, SVHN_On_Memory

from nose.plugins.skip import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestSVHN(unittest.TestCase):
    #WARN: SVHN tries to overwrite files as part of __init__.
    def setUp(self):
        try:
            self.train = SVHN(which_set='train')
            self.test = SVHN(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    def test_get_test_set(self):
        try:
            test_from_train = self.train.get_test_set()
            self.assertTrue(numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix()))
            self.assertTrue(numpy.all(test_from_train.get_targets() == self.test.get_targets()))
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

class TestSVHN_On_Memory(unittest.TestCase):
    def setUp(self):
        try:
            self.train = SVHN_On_Memory(which_set='train')
            self.test = SVHN_On_Memory(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    def test_get_test_set(self):
        try:
            test_from_train = self.train.get_test_set()
            self.assertTrue(numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix()))
            self.assertTrue(numpy.all(test_from_train.get_targets() == self.test.get_targets()))
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

#TODO: test the @staticmethod make_data
