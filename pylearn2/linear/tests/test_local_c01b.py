import theano
from theano import tensor
import numpy
from pylearn2.linear.local_c01b import Local, make_random_local
from pylearn2.utils import sharedX
from pylearn2.testing.skip import skip_if_no_gpu
import unittest


class TestConv2DC01b(unittest.TestCase):
    """
    Test for local receptive fields
    """
    def setUp(self):
        """
        Set up a test image and filter to re-use
        """
        skip_if_no_gpu()
        self.image = \
            numpy.random.rand(16, 3, 3, 1).astype(theano.config.floatX)
        self.image_tensor = tensor.tensor4()
        self.filters_values = numpy.ones(
            (2, 2, 16, 2, 2, 1, 16), dtype=theano.config.floatX
        )
        self.filters = sharedX(self.filters_values, name='filters')
        self.local = Local(self.filters, (3, 3), 1)

    def test_get_params(self):
        """
        Check whether the local receptive field has stored the correct filters
        """
        assert self.local.get_params() == [self.filters]

    def test_lmul(self):
        """
        Make sure the shape of the output is correct
        """
        f = theano.function([self.image_tensor],
                            self.local.lmul(self.image_tensor))
        assert f(self.image).shape == (16, 2, 2, 1)

    def test_make_random_local(self):
        """
        Create random local receptive fields and check whether they can be
        applied and give a sensible output shape
        """
        local = make_random_local(1, 16, ('c', 0, 1, 'b'), 1, (3, 3),
                                  16, ('c', 0, 1, 'b'), (2, 2))
        f = theano.function([self.image_tensor],
                            local.lmul(self.image_tensor))
        assert f(self.image).shape == (16, 2, 2, 1)
