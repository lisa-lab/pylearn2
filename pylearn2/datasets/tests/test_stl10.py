import unittest
from pylearn2.datasets.stl10 import STL10, restrict

from unittest import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestSTL10(unittest.TestCase):
    def setUp(self):
        try:
            self.train = STL10(which_set='train')
            self.test = STL10(which_set='test') 
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    def test_restrict(self):
        #restrict provides a series of tests that are comprehensive enough
        #right now.
        restrict(self.train, 0)
        restrict(self.test, 0)
