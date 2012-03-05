
import unittest

import numpy

from .linear import LinearTransform
from .linear import dot
from .linear import dot_shape
from .linear import dot_shape_from_shape


class ReshapeBase(LinearTransform):
    def __init__(self, from_shp, to_shp):
        LinearTransform.__init__(self, [])
        self._from_shp = from_shp
        self._to_shp = to_shp

    def row_shape(self):
        return self._to_shp

    def col_shape(self):
        return self._from_shp


class ReshapeL(ReshapeBase):

    def lmul(self, x):
        RR, CC = self.split_left_shape(x.shape, False)
        return x.reshape(RR + self._to_shp)

    def rmul(self, x):
        RR, CC = self.split_right_shape(x.shape, False)
        return x.reshape(self._from_shp + CC)


class ReshapeR(ReshapeBase):

    def lmul_T(self, x):
        CC, RR = self.split_right_shape(x.shape, True)
        return x.reshape(CC + self._from_shp)

    def rmul_T(self, x):
        CC, RR = self.split_left_shape(x.shape, True)
        return x.reshape(self._to_shp + RR)


class NumericSelfTestMixin(object):
    """
    Generic tests that assert the self-consistency of LinearTransform
    implementations that operate on numpy arrays.

    """

    def test_shape_xl_A(self):
        xl_A = dot(self.xl, self.A)
        assert xl_A.shape == dot_shape(self.xl, self.A)

    def test_shape_A_xr(self):
        A_xr = dot(self.A, self.xr)
        A_xr_shape = dot_shape(self.A, self.xr)
        assert A_xr.shape == A_xr_shape, (A_xr.shape, A_xr_shape)

    def test_shape_xrT_AT(self):
        # dot (xr.T, A.T)
        AT = self.A.T
        xrT_AT = dot(AT.transpose_left(self.xr, T=True), AT)
        assert xrT_AT.shape == dot_shape_from_shape(
                AT.transpose_left_shape(self.xr.shape, T=True), AT)

    def test_shape_AT_xlT(self):
        # dot (A.T, xl.T)
        AT = self.A.T
        AT_xlT = dot(AT,
                AT.transpose_right(self.xl, T=True))
        AT_xlt_shape = dot_shape_from_shape(AT,
                AT.transpose_right_shape(self.xl.shape, T=True))
        assert AT_xlT.shape == AT_xlt_shape, (AT_xlT.shape, AT_xlt_shape)


class TestReshapeL(NumericSelfTestMixin):
    def setUp(self):
        self.xl = numpy.random.randn(4, 3, 2)  # for left-mul
        self.xr = numpy.random.randn(6, 5)     # for right-mul
        self.A = ReshapeL((3, 2), (6,))
        self.xl_A_shape = (4, 6)

    def test_xl_A_value(self):
        xl_A = dot(self.xl, self.A)
        assert numpy.all(xl_A == self.xl.reshape(xl_A.shape))

class TestReshapeR(NumericSelfTestMixin):
    def setUp(self):
        self.xl = numpy.random.randn(4, 3, 2)  # for left-mul
        self.xr = numpy.random.randn(6, 5)     # for right-mul
        self.A = ReshapeR((3, 2), (6,))
        self.xl_A_shape = (4, 6)

    def test_xl_A_value(self):
        xl_A = dot(self.xl, self.A)
        assert numpy.all(xl_A == self.xl.reshape(xl_A.shape))

