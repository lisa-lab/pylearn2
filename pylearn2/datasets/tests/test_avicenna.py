import numpy
import unittest
from numpy.testing.decorators import knownfailureif
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError
from nose.plugins.skip import SkipTest

RELIES_ON_PYLEARN_ONE = True

class TestAvicenna(unittest.TestCase):
    @knownfailureif(RELIES_ON_PYLEARN_ONE)
    def setUp(self):
        try:
            from pylearn2.datasets.avicenna import Avicenna
            self.train = Avicenna(which_set='train', standardize=False)
            self.test = Avicenna(which_set='test', standardize=False)

            self.train_std = Avicenna(which_set='train', standardize=True)
            self.test_std = Avicenna(which_set='test', standardize=True)
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

        #Avicenna doesn't define any (real) method of its own, this
        #simply tests if data can be loaded without error.
