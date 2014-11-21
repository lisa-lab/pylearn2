import unittest

import numpy

import theano
from theano.tests.unittest_tools import verify_grad

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

def rand(shp, dtype):
    return numpy.random.rand(*shp).astype(dtype)

def assert_linear(f, pt, mode=None):
    t = theano.tensor.scalar(dtype=pt.dtype)
    ptlike = theano.shared(rand(
        pt.get_value(borrow=True).shape,
        dtype=pt.dtype))
    out = f(pt)
    out2 = f(pt * t)
    out3 = f(ptlike) + out
    out4 = f(pt + ptlike)

    f = theano.function([t], [out * t, out2, out3, out4],
            allow_input_downcast=True,
            mode=mode)
    outval, out2val, out3val, out4val = f(3.6)
    assert numpy.allclose(outval, out2val)
    assert numpy.allclose(out3val, out4val)


class TestFilterActs(unittest.TestCase):
    # 2 4x4 greyscale images
    ishape = (1, 1, 4, 4, 2)
    # 5 3x3 filters at each location in a 2x2 grid
    fshape = (2, 2, 1, 3, 3, 1, 5)
    module_stride = 1
    dtype = 'float64'
    # step size for numeric gradient, None is the default
    eps = None
    mode = theano.compile.get_default_mode()

    def function(self, inputs, outputs):
        return theano.function(inputs, outputs, mode=self.mode)

    def setUp(self):
        self.op = FilterActs(self.module_stride)
        self.s_images = theano.shared(rand(self.ishape, self.dtype),
                name = 's_images')
        self.s_filters = theano.shared(
                rand(self.fshape, self.dtype),
                name = 's_filters')

    def test_type(self):
        out = self.op(self.s_images, self.s_filters)
        assert out.dtype == self.dtype
        assert out.ndim == 5

        f = self.function([], out)
        outval = f()
        assert len(outval.shape) == len(self.ishape)
        assert outval.dtype == self.s_images.get_value(borrow=True).dtype

    def test_linearity_images(self):
        assert_linear(
                lambda imgs: self.op(imgs, self.s_filters),
                self.s_images,
                mode=self.mode)

    def test_linearity_filters(self):
        assert_linear(
                lambda fts: self.op(self.s_images, fts),
                self.s_filters,
                mode=self.mode)

    def test_shape(self):
        out = self.op(self.s_images, self.s_filters)
        f = self.function([], out)
        outval = f()
        assert outval.shape == (self.fshape[-2],
                self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_grad_left(self):
        # test only the left so that the right can be a shared variable,
        # and then TestGpuFilterActs can use a gpu-allocated shared var
        # instead.
        def left_op(imgs):
            return self.op(imgs, self.s_filters)

        verify_grad(left_op, [self.s_images.get_value()],
                    mode=self.mode, eps=self.eps)

    def test_grad_right(self):
        # test only the right so that the left can be a shared variable,
        # and then TestGpuFilterActs can use a gpu-allocated shared var
        # instead.
        def right_op(filters):
            rval =  self.op(self.s_images, filters)
            rval.name = 'right_op(%s, %s)' % (self.s_images.name,
                    filters.name)
            assert rval.dtype == filters.dtype
            return rval

        verify_grad(right_op, [self.s_filters.get_value()],
                    mode=self.mode, eps=self.eps)

    def test_dtype_mismatch(self):
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_images, 'float32'),
                theano.tensor.cast(self.s_filters, 'float64'))
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_images, 'float64'),
                theano.tensor.cast(self.s_filters, 'float32'))

    def test_op_eq(self):
        assert FilterActs(1) == FilterActs(1)
        assert not (FilterActs(1) != FilterActs(1))
        assert (FilterActs(2) != FilterActs(1))
        assert FilterActs(1) != None


class TestFilterActsF32(TestFilterActs):
    dtype = 'float32'
    eps = 1e-3


class TestWeightActs(unittest.TestCase):
    # 1 5x5 6-channel image (2 groups of 3 channels)
    ishape = (6, 3, 5, 5, 1)
    hshape = (6, 4, 2, 2, 1)
    fshape = (2, 2, 3, 2, 2, 6, 4)
    module_stride = 2
    dtype = 'float64'
    # step size for numeric gradient, None is the default
    eps = None

    frows = property(lambda s: s.fshape[3])
    fcols = property(lambda s: s.fshape[4])

    def setUp(self):
        self.op = WeightActs(self.module_stride)
        self.s_images = theano.shared(rand(self.ishape, self.dtype))
        self.s_hidacts = theano.shared(rand(self.hshape, self.dtype))

    def test_type(self):
        out = self.op(self.s_images, self.s_hidacts, self.frows, self.fcols)
        assert out.dtype == self.dtype
        assert out.ndim == 7
        f = theano.function([], out)
        outval = f()
        assert outval.shape == self.fshape
        assert outval.dtype == self.dtype

    def test_linearity_images(self):
        def f(images):
            return self.op(images, self.s_hidacts, self.frows, self.fcols)
        assert_linear(f, self.s_images)

    def test_linearity_hidacts(self):
        def f(hidacts):
            return self.op(self.s_images, hidacts, self.frows, self.fcols)
        assert_linear(f, self.s_hidacts)

    def test_grad(self):
        def op2(imgs, hids):
            return self.op(imgs, hids, self.frows, self.fcols)

        verify_grad(op2,
                    [self.s_images.get_value(),
                     self.s_hidacts.get_value()],
                    eps=self.eps)

    def test_dtype_mismatch(self):
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_images, 'float32'),
                theano.tensor.cast(self.s_hidacts, 'float64'),
                self.frows, self.fcols)
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_images, 'float64'),
                theano.tensor.cast(self.s_hidacts, 'float32'),
                self.frows, self.fcols)


class TestImgActs(unittest.TestCase):
    # 1 5x5 6-channel image (2 groups of 3 channels)
    ishape = (6, 3, 5, 5, 2)
    hshape = (6, 4, 3, 3, 2)
    fshape = (3, 3, 3, 2, 2, 6, 4)
    module_stride = 1
    dtype = 'float64'
    # step size for numeric gradient, None is the default
    eps = None

    #frows = property(lambda s: s.fshape[3])
    #fcols = property(lambda s: s.fshape[4])
    irows = property(lambda s: s.ishape[2])
    icols = property(lambda s: s.ishape[3])

    def setUp(self):
        self.op = ImgActs(module_stride=self.module_stride)
        self.s_filters = theano.shared(rand(self.fshape, self.dtype))
        self.s_hidacts = theano.shared(rand(self.hshape, self.dtype))

    def test_type(self):
        out = self.op(self.s_filters, self.s_hidacts, self.irows, self.icols)
        assert out.dtype == self.dtype
        assert out.ndim == 5
        f = theano.function([], out)
        outval = f()
        assert outval.shape == self.ishape
        assert outval.dtype == self.dtype

    def test_linearity_filters(self):
        def f(filts):
            return self.op(filts, self.s_hidacts, self.irows, self.icols)
        assert_linear(f, self.s_filters)

    def test_linearity_hidacts(self):
        def f(hidacts):
            return self.op(self.s_filters, hidacts, self.irows, self.icols)
        assert_linear(f, self.s_hidacts)

    def test_grad(self):
        def op2(imgs, hids):
            return self.op(imgs, hids, self.irows, self.icols)

        verify_grad(op2,
                    [self.s_filters.get_value(),
                     self.s_hidacts.get_value()],
                    eps=self.eps)

    def test_dtype_mismatch(self):
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_filters, 'float32'),
                theano.tensor.cast(self.s_hidacts, 'float64'),
                self.irows, self.icols)
        self.assertRaises(TypeError,
                self.op,
                theano.tensor.cast(self.s_filters, 'float64'),
                theano.tensor.cast(self.s_hidacts, 'float32'),
                self.irows, self.icols)




