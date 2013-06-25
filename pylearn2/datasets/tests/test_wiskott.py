import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.wiskott import Wiskott

class TestWiskott(unittest.TestCase):
    def setUp(self):
        skip_if_no_data('wiskott')
        self.dataset = Wiskott()
        #wiskott doesn't define any method of its own
    
