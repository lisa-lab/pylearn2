"""
Tests for the Cudnn code.
"""

__author__ = "Francesco Visin"
__license__ = "3-clause BSD"
__credits__ = "Francesco Visin"
__maintainer__ = "Lisa Lab"

import theano
from theano import tensor
from theano.sandbox.cuda.dnn import dnn_available

from pylearn2.linear.conv2d import Conv2D
from pylearn2.linear.cudnn2d import Cudnn2D, make_random_conv2D
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.testing.skip import skip_if_no_gpu

import unittest
from nose.plugins.skip import SkipTest

import numpy as np


class TestCudnn(unittest.TestCase):
    """
    Tests for the Cudnn code.

    Parameters
    ----------
    Refer to unittest.TestCase.

    """
    def setUp(self):
        """
        Set up a test image and filter to re-use.
        """
        skip_if_no_gpu()
        if not dnn_available():
            raise SkipTest('Skipping tests cause cudnn is not available')
        self.orig_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        self.image = np.random.rand(1, 1, 3, 3).astype(theano.config.floatX)
        self.image_tensor = tensor.tensor4()
        self.input_space = Conv2DSpace((3, 3), 1, axes=('b', 'c', 0, 1))
        self.filters_values = np.ones(
            (1, 1, 2, 2), dtype=theano.config.floatX
        )
        self.filters = sharedX(self.filters_values, name='filters')
        self.batch_size = 1

        self.cudnn2d = Cudnn2D(self.filters, self.batch_size, self.input_space)

    def tearDown(self):
        """
        After test clean up.
        """
        theano.config.floatX = self.orig_floatX

    def test_value_errors(self):
        """
        Check correct errors are raised when bad input is given.
        """
        with self.assertRaises(AssertionError):
            Cudnn2D(filters=self.filters, batch_size=-1,
                    input_space=self.input_space)

    def test_get_params(self):
        """
        Check whether the cudnn has stored the correct filters.
        """
        self.assertEqual(self.cudnn2d.get_params(), [self.filters])

    def test_get_weights_topo(self):
        """
        Check whether the cudnn has stored the correct filters.
        """
        self.assertTrue(np.all(
            self.cudnn2d.get_weights_topo(borrow=True) ==
            np.transpose(self.filters.get_value(borrow=True), (0, 2, 3, 1))))

    def test_lmul(self):
        """
        Use conv2D to check whether the convolution worked correctly.
        """
        conv2d = Conv2D(self.filters, self.batch_size, self.input_space,
                        output_axes=('b', 'c', 0, 1),)
        f_co = theano.function([self.image_tensor],
                               conv2d.lmul(self.image_tensor))
        f_cu = theano.function([self.image_tensor],
                               self.cudnn2d.lmul(self.image_tensor))
        self.assertTrue(np.allclose(f_co(self.image), f_cu(self.image)))

    def test_set_batch_size(self):
        """
        Make sure that setting the batch size actually changes the property.
        """
        img_shape = self.cudnn2d._img_shape
        self.cudnn2d.set_batch_size(self.batch_size + 10)
        np.testing.assert_equal(self.cudnn2d._img_shape[0],
                                self.batch_size + 10)
        np.testing.assert_equal(self.cudnn2d._img_shape[1:], img_shape[1:])

    def test_axes(self):
        """
        Test different output axes.

        Use different output axes and see whether the output is what we
        expect.
        """
        default_axes = ('b', 'c', 0, 1)
        axes = (0, 'b', 1, 'c')
        another_axes = (0, 1, 'c', 'b')
        # 1, 3, 0, 2
        map_to_default = tuple(axes.index(axis) for axis in default_axes)
        # 2, 0, 3, 1
        map_to_another_axes = tuple(default_axes.index(axis) for
                                    axis in another_axes)
        input_space = Conv2DSpace((3, 3), num_channels=1, axes=another_axes)
        # Apply cudnn2d with `axes` as output_axes
        cudnn2d = Cudnn2D(self.filters, 1, input_space, output_axes=axes)
        f = theano.function([self.image_tensor],
                            cudnn2d.lmul(self.image_tensor))
        # Apply cudnn2d with default axes
        f_def = theano.function([self.image_tensor],
                                self.cudnn2d.lmul(self.image_tensor))

        # Apply f on the `another_axes`-shaped image
        output = f(np.transpose(self.image, map_to_another_axes))
        # Apply f_def on self.image (b,c,0,1)
        output_def = np.array(f_def(self.image))
        # transpose output to def
        output = np.transpose(output, map_to_default)

        np.testing.assert_allclose(output_def, output)
        np.testing.assert_equal(output_def.shape, output.shape)

    def test_channels(self):
        """
        Go from 2 to 3 channels and see whether the shape is correct.
        """
        input_space = Conv2DSpace((3, 3), num_channels=3)
        filters_values = np.ones(
            (2, 3, 2, 2), dtype=theano.config.floatX
        )
        filters = sharedX(filters_values)
        image = np.random.rand(1, 3, 3, 3).astype(theano.config.floatX)
        cudnn2d = Cudnn2D(filters, 1, input_space)
        f = theano.function([self.image_tensor],
                            cudnn2d.lmul(self.image_tensor))
        assert f(image).shape == (1, 2, 2, 2)

    def test_make_random_conv2D(self):
        """
        Test a random convolution.

        Create a random convolution and check whether the shape, axes and
        input space are all what we expect.
        """
        output_space = Conv2DSpace((2, 2), 1)
        cudnn2d = make_random_conv2D(1, self.input_space, output_space,
                                     (2, 2), 1)
        f = theano.function([self.image_tensor],
                            cudnn2d.lmul(self.image_tensor))
        assert f(self.image).shape == (1, 2, 2, 1)
        assert cudnn2d._input_space == self.input_space
        assert cudnn2d._output_axes == output_space.axes
