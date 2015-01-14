"""
Testing class that simply checks to see if the class is
loadable
"""
from pylearn2.datasets.mnist import MNIST_rotated_background
from pylearn2.datasets.tests.test_mnist import TestMNIST
from pylearn2.testing.skip import skip_if_no_data


class TestMNIST_rotated(TestMNIST):
    """
    Parameters
    ----------
    None

    Notes
    -----
    Testing class that simply checks to see if the rotated mnist is
    loadable
    """
    def setUp(self):
        """
        Attempts to load train and test
        """
        skip_if_no_data()
        self.train = MNIST_rotated_background(which_set='train')
        self.test = MNIST_rotated_background(which_set='test')
