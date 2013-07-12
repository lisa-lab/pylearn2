import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.iris import Iris

from unittest import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError



class TestIris(unittest.TestCase):
    def setUp(self):
        try:
            self.dataset = Iris()
        except NoDataPathError, NotInstalledError:
            raise SkipTest()

        #nothing else defined in Iris
