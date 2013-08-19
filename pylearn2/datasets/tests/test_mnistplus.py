import numpy
from pylearn2.testing.skip import skip_if_no_data
import unittest
from pylearn2.datasets.mnistplus import MNISTPlus

from nose.plugins.skip import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestMNISTPlus(unittest.TestCase):
    def setUp(self):
        try:
            self.train = MNISTPlus(which_set='train')
            self.test = MNISTPlus(which_set='test')
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

        #MNISTPlus actually doesn't really implement any method of its own,
        #so this test simply verifies that it can load data without errors.
