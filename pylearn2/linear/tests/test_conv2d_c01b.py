import theano
from theano import tensor
from theano.compat.six.moves import xrange
import numpy
from pylearn2.linear.conv2d_c01b import (Conv2D, make_random_conv2D,
    make_sparse_random_conv2D, setup_detector_layer_c01b)
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.testing.skip import skip_if_no_gpu
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.models.mlp import MLP
skip_if_no_gpu()
import unittest
try:
    scipy_available = True
    import scipy.ndimage
except ImportError:
    scipy_available = False


class TestConv2DC01b(unittest.TestCase):
    """
    Tests for Alex Krizhevsky's Conv2D code
    """
    def setUp(self):
        """
        Set up a test image and filter to re-use
        """
        self.orig_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        theano.sandbox.cuda.use('gpu')
        self.image = \
            numpy.random.rand(16, 3, 3, 1).astype(theano.config.floatX)
        self.image_tensor = tensor.tensor4()
        self.filters_values = numpy.random.rand(
            16, 2, 2, 32).astype(theano.config.floatX)
        self.filters = sharedX(self.filters_values, name='filters')
        self.conv2d = Conv2D(self.filters)

    def tearDown(self):
        theano.config.floatX = self.orig_floatX
        theano.sandbox.cuda.unuse()

    def scipy_conv_c01b(self, images, filters):
        """
        Emulate c01b convolution with scipy
        """
        assert images.ndim == 4
        assert filters.ndim == 4
        in_chans, rows, cols, bs = images.shape
        in_chans_, rows_, cols_, out_chans = filters.shape
        assert in_chans_ == in_chans

        out_bc01 = [
            [sum(scipy.ndimage.filters.convolve(images[c, :, :, b],
                                                filters[c, ::-1, ::-1, i])
                 for c in xrange(in_chans))
             for i in xrange(out_chans)]
            for b in xrange(bs)]
        out_c01b = numpy.array(out_bc01).transpose(1, 2, 3, 0)
        return out_c01b

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
            self.assertTrue(
                numpy.allclose(
                    f(self.image),
                    self.scipy_conv_c01b(self.image,
                                         self.filters_values)[:, :2, :2, :]))

    def test_lmul_T(self):
        """
        Check whether this function outputs the right shape
        """
        conv2d = self.conv2d.lmul(self.image_tensor)
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul_T(conv2d))
        assert f(self.image).shape == self.image.shape

    def test_axes(self):
        """
        Use custom output axes and check whether it worked
        """
        default_axes = ('c', 0, 1, 'b')
        axes = (0, 'b', 1, 'c')
        mapping = tuple(axes.index(axis) for axis in default_axes)
        conv2d = Conv2D(self.filters, output_axes=axes)
        f_axes = theano.function([self.image_tensor],
                                 conv2d.lmul(self.image_tensor))
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul(self.image_tensor))
        output_axes = f_axes(self.image)
        output = f(self.image)
        output_axes = numpy.transpose(output_axes, mapping)
        numpy.testing.assert_allclose(output, output_axes)
        assert output.shape == output_axes.shape

    def test_channels(self):
        """
        Go from 32 to 16 channels and see whether that works without error
        """
        filters_values = numpy.ones(
            (32, 2, 2, 16), dtype=theano.config.floatX
        )
        filters = sharedX(filters_values)
        image = numpy.random.rand(32, 3, 3, 1).astype(theano.config.floatX)
        conv2d = Conv2D(filters)
        f = theano.function([self.image_tensor],
                            conv2d.lmul(self.image_tensor))
        assert f(image).shape == (16, 2, 2, 1)

    def test_make_random_conv2D(self):
        """
        Make random filters
        """
        default_axes = ('c', 0, 1, 'b')
        conv2d = make_random_conv2D(1, 16, default_axes, default_axes,
                                    16, (2, 2))
        f = theano.function([self.image_tensor],
                            conv2d.lmul(self.image_tensor))
        assert f(self.image).shape == (16, 2, 2, 1)
        assert conv2d.output_axes == default_axes

    def test_make_sparse_random_conv2D(self):
        """
        Make random sparse filters, count whether the number of
        non-zero elements is sensible
        """
        axes = ('c', 0, 1, 'b')
        input_space = Conv2DSpace((3, 3), 16, axes=axes)
        output_space = Conv2DSpace((3, 3), 16, axes=axes)
        num_nonzero = 2
        kernel_shape = (2, 2)

        conv2d = make_sparse_random_conv2D(num_nonzero, input_space,
                                           output_space, kernel_shape)
        f = theano.function([self.image_tensor],
                            conv2d.lmul(self.image_tensor))
        assert f(self.image).shape == (16, 2, 2, 1)
        assert conv2d.output_axes == axes
        assert numpy.count_nonzero(conv2d._filters.get_value()) >= 32

    def test_setup_detector_layer_c01b(self):
        """
        Very basic test to see whether a detector layer can be set up
        without error. Not checking much for the actual output.
        """
        axes = ('c', 0, 1, 'b')
        layer = MaxoutConvC01B(16, 2, (2, 2), (2, 2),
                               (1, 1), 'maxout', irange=1.)
        input_space = Conv2DSpace((3, 3), 16, axes=axes)
        MLP(layers=[layer], input_space=input_space)
        layer.set_input_space(input_space)
        assert isinstance(layer.input_space, Conv2DSpace)
        input = theano.tensor.tensor4()
        f = theano.function([input], layer.fprop(input))
        f(numpy.random.rand(16, 3, 3, 1).astype(theano.config.floatX))
