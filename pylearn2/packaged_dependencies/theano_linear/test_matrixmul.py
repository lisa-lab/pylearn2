
import unittest

import numpy

import theano
from theano import tensor

from .linear import LinearTransform
from .linear import dot
from .linear import dot_shape
from .linear import dot_shape_from_shape
from .matrixmul import MatrixMul


def assert_compute_equal(outputs, inputs=[]):
    outputs = map(tensor.as_tensor_variable, outputs)
    f = theano.function(inputs, outputs)
    outvals = f()
    assert all(numpy.all(outvals[i] == outvals[0])
            for i in range(1, len(outvals))), (outvals)

class SymbolicSelfTestMixin(object):
    """
    Generic tests that assert the self-consistency of LinearTransform
    implementations that operate on Theano variables.

    """

    def test_shape_xl_A(self):
        xl_A = dot(self.xl, self.A)
        assert_compute_equal([xl_A.shape, dot_shape(self.xl, self.A)])

    def test_shape_A_xr(self):
        A_xr = dot(self.A, self.xr)
        assert_compute_equal([A_xr.shape, dot_shape(self.A, self.xr)])

    def test_shape_xrT_AT(self):
        # dot (xr.T, A.T)
        AT = self.A.T
        xrT_AT = dot(AT.transpose_left(self.xr, T=True), AT)
        assert_compute_equal([
                xrT_AT.shape,
                dot_shape_from_shape(
                    AT.transpose_left_shape(tuple(self.xr.shape), T=True), AT)])

    def test_shape_AT_xlT(self):
        # dot (A.T, xl.T)
        AT = self.A.T
        AT_xlT = dot(AT,
                AT.transpose_right(self.xl, T=True))
        AT_xlt_shape = dot_shape_from_shape(AT,
                AT.transpose_right_shape(tuple(self.xl.shape), T=True))
        assert_compute_equal([
                AT_xlT.shape,
                AT_xlt_shape])


class TestMatrixMul(unittest.TestCase, SymbolicSelfTestMixin):
    def setUp(self):
        self.xlval = 0.5 + numpy.random.randn(4, 3, 2)
        self.xrval = 0.5 + numpy.random.randn(7, 5)
        self.Wval = numpy.random.rand(6, 7) + 0.5
        self.xl = theano.shared(self.xlval)
        self.xr = theano.shared(self.xrval)
        self.W = theano.shared(self.Wval)
        self.A = MatrixMul(self.W, col_shape = (3, 2))

    def test_xl_A_value(self):
        xl_A = numpy.dot(self.xlval.reshape(4, 6), self.Wval)
        assert_compute_equal([xl_A, dot(self.xl, self.A)])

    def test_A_xr_value(self):
        val = numpy.dot(self.Wval, self.xrval).reshape(3, 2, 5)
        assert_compute_equal([val, dot(self.A, self.xr)])

    def test_AT_xlT_value(self):
        val = numpy.dot(self.Wval.T,
                self.xlval.transpose(1, 2, 0).reshape(-1, 4))
        assert_compute_equal([val,
            dot(self.A.T, self.A.transpose_left(self.xl, T=False))])

    def test_xrT_AT(self):
        val = numpy.dot(
                self.xrval.transpose(),
                self.Wval.T).reshape(5, 3, 2)
        assert_compute_equal([val,
            dot(self.A.transpose_right(self.xr, T=False), self.A.T)])
