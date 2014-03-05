import theano
from theano import tensor
import numpy
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace
import unittest
try:
    scipy_available = True
    import scipy.ndimage
except:
    scipy_available = False


class TestConv2D(unittest.TestCase):
    def setUp(self):
        self.image = numpy.random.randint(0, 10,
                                          (1, 3, 3, 1)).astype('float32')
        self.image_tensor = tensor.tensor4()
        self.input_space = Conv2DSpace((3, 3), 1)
        self.filters_values = numpy.ones((1, 1, 2, 2)).astype('float32')
        self.filters = theano.shared(value=self.filters_values, name='filters')
        self.conv2d = Conv2D(self.filters, 1, self.input_space)

    def test_value_errors(self):
        bad_filters = theano.shared(value=numpy.zeros((1, 3, 2)))
        with self.assertRaises(TypeError):
            Conv2D(bad_filters, 1, self.input_space)
        with self.assertRaises(AssertionError):
            Conv2D(self.filters, 0, self.input_space)

    def test_get_params(self):
        assert self.conv2d.get_params() == [self.filters]

    def test_lmul(self):
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul(self.image_tensor))
        if scipy_available:
            numpy.allclose(
                f(self.image).reshape((2, 2)),
                scipy.ndimage.filters.convolve(
                    self.image.reshape((3, 3)),
                    self.filters_values.reshape((2, 2))
                )
            )

    def test_lmul_T(self):
        f = theano.function([self.image_tensor],
                            self.conv2d.lmul_T(self.image_tensor))
        assert f(self.image).shape == (1, 4, 4, 1)

    def test_lmul_T_sq(self):
        conv2d_sq = Conv2D(
            theano.shared(value=numpy.square(self.filters_values)),
            1, self.input_space
        )
        f = theano.function([self.image_tensor],
                            conv2d_sq.lmul_T(self.image_tensor))
        f2 = theano.function([self.image_tensor],
                             self.conv2d.lmul_sq_T(self.image_tensor))
        numpy.testing.assert_allclose(f(self.image), f2(self.image))

    def test_set_batch_size(self):
        cur_img_shape = self.conv2d._img_shape
        cur_batch_size = self.conv2d._img_shape[0]
        self.conv2d.set_batch_size(cur_batch_size + 10)
        assert self.conv2d._img_shape[0] == cur_batch_size + 10
        assert self.conv2d._img_shape[1:] == cur_img_shape[1:]

    def test_axes(self):
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
