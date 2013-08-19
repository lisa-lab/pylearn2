import unittest
from pylearn2.datasets.wiskott import Wiskott

from nose.plugins.skip import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestWiskott(unittest.TestCase):
    def setUp(self):
        try:
            self.dataset = Wiskott()
            #wiskott doesn't define any method of its own
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

