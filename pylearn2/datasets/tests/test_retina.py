import numpy
import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.retina import * #XXX

class TestRetina(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        #TODO: how to test?
