import numpy
import unittest
from pylearn2.datasets.ocr import OCR

from unittest import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestOCR(unittest.TestCase):
    def setUp(self):
        #TODO: test axes, test one_hot
        try:
            self.train = OCR(which_set='train')
            self.test = OCR(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    def get_test_set(self):
        try:
            test_from_train = self.train.get_test_set()
            self.assertTrue(numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix()))
            self.assertTrue(numpy.all(test_from_train.get_targets() == self.test.get_targets()))
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

