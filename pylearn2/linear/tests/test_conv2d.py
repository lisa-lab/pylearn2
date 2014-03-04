import theano
from theano import tensor
import numpy
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace
import unittest

class TestConv2D(unittest.TestCase):
    def setUp(self):
        image = numpy.random.rand(3, 3)
        self.input_space = Conv2DSpace((3, 3), 1)
        filters_values = numpy.zeros((1, 1, 2, 2))
        filters = theano.shared(value=filters_values, name='filters')
        self.conv2d = Conv2D(filters, 1, self.input_space)

    def test_value_errors(self):
        filter = theano.shared(value=numpy.zeros((1, 2, 3, 4)))
        bad_filter = theano.shared(value=numpy.zeros((1, 3, 2)))
        self.assertRaises(TypeError, Conv2D, (bad_filter, 1,
                                              self.input_space))
        self.assertRaises(AssertionError, Conv2D, (filter, 0,
                                                   self.input_space))

