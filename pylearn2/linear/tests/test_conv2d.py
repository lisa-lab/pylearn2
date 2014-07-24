import theano
from theano import tensor
import numpy
from pylearn2.linear.conv2d import Conv2D, make_random_conv2D
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
import unittest
try:
    scipy_available = True
    import scipy.ndimage
except ImportError:
    scipy_available = False


class TestConv2D(unittest.TestCase):
    """
    Tests for Conv2D code
    """
    def setUp(self):
        """
        Set up a test image and filter to re-use
        """
        self.image = numpy.random.rand(1, 3, 3, 1).astype(theano.config.floatX)
        self.image_tensor = tensor.tensor4()
        self.input_space = Conv2DSpace((3, 3), 1)
        self.filters_values = numpy.ones(
            (1, 1, 2, 2), dtype=theano.config.floatX
        )
        self.filters = sharedX(self.filters_values, name='filters')
        self.conv2d = Conv2D(self.filters, 1, self.input_space)

    def test_value_errors(self):
        """
        Check correct errors are raised when bad input is given
        """
        bad_filters = sharedX(numpy.zeros((1, 3, 2)))
        self.assertRaises(ValueError, Conv2D, bad_filters, 1, self.input_space)
        self.assertRaises(AssertionError, Conv2D, self.filters, 0,
                          self.input_space)

    def test_get_params(self):
        """
        Check whether the conv2d has stored the correct filters
        """
        assert self.conv2d.get_params() == [self.filters]

    def test_lmul(self):
        """
        Use SciPy's ndimage to check whether the convolution worked
        correctly
        """
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul(self.image_tensor))
        if scipy_available:
            numpy.allclose(
                f(self.image).reshape((2, 2)),
                scipy.ndimage.filters.convolve(
                    self.image.reshape((3, 3)),
                    self.filters_values.reshape((2, 2))
                )[:2, :2]
            )

    def test_lmul_T(self):
        """
        Check whether this function outputs the right shape
        """
        conv2d = self.conv2d.lmul(self.image_tensor)
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul_T(conv2d))
        assert f(self.image).shape == self.image.shape

    def test_lmul_sq_T(self):
        """
        Check whether this function outputs the same values as when
        taking the square manually
        """
        conv2d_sq = Conv2D(sharedX(numpy.square(self.filters_values)),
            1, self.input_space
        ).lmul(self.image_tensor)
        conv2d = self.conv2d.lmul(self.image_tensor)
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul_T(conv2d_sq))
        f2 = theano.function([self.image_tensor],
                             self.conv2d.lmul_sq_T(conv2d))

        numpy.testing.assert_allclose(f(self.image), f2(self.image))

    def test_set_batch_size(self):
        """
        Make sure that setting the batch size actually changes the property
        """
        cur_img_shape = self.conv2d._img_shape
        cur_batch_size = self.conv2d._img_shape[0]
        self.conv2d.set_batch_size(cur_batch_size + 10)
        assert self.conv2d._img_shape[0] == cur_batch_size + 10
        assert self.conv2d._img_shape[1:] == cur_img_shape[1:]

    def test_axes(self):
        """
        Use different output axes and see whether the output is what we
        expect
        """
        default_axes = ('b', 0, 1, 'c')
        axes = (0, 'b', 1, 'c')
        mapping = tuple(axes.index(axis) for axis in default_axes)
        input_space = Conv2DSpace((3, 3), num_channels=1, axes=axes)
        conv2d = Conv2D(self.filters, 1, input_space, output_axes=axes)
        f_axes = theano.function([self.image_tensor],
                                 conv2d.lmul(self.image_tensor))
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul(self.image_tensor))
        output_axes = f_axes(numpy.transpose(self.image, mapping))
        output = f(self.image)
        output_axes = numpy.transpose(output_axes, mapping)
        numpy.testing.assert_allclose(output, output_axes)
        assert output.shape == output_axes.shape

    def test_channels(self):
        """
        Go from 2 to 3 channels and see whether the shape is correct
        """
        input_space = Conv2DSpace((3, 3), num_channels=3)
        filters_values = numpy.ones(
            (2, 3, 2, 2), dtype=theano.config.floatX
        )
        filters = sharedX(filters_values)
        image = numpy.random.rand(1, 3, 3, 3).astype(theano.config.floatX)
        conv2d = Conv2D(filters, 1, input_space)
        f = theano.function([self.image_tensor],
                            conv2d.lmul(self.image_tensor))
        assert f(image).shape == (1, 2, 2, 2)

    def test_make_random_conv2D(self):
        """
        Create a random convolution and check whether the shape, axes and
        input space are all what we expect
        """
        output_space = Conv2DSpace((2, 2), 1)
        conv2d = make_random_conv2D(1, self.input_space, output_space,
                                    (2, 2), 1)
        f = theano.function([self.image_tensor],
                            conv2d.lmul(self.image_tensor))
        assert f(self.image).shape == (1, 2, 2, 1)
        assert conv2d.input_space == self.input_space
        assert conv2d.output_axes == output_space.axes
