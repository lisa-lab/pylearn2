
import unittest

import numpy

import theano

from localdot import LocalDot

from ..test_matrixmul import SymbolicSelfTestMixin


class TestLocalDot32x32(unittest.TestCase, SymbolicSelfTestMixin):
    channels = 3
    bsize = 10     # batch size
    imshp = (32, 32)
    ksize = 5
    nkern_per_group = 16
    subsample_stride = 1
    ngroups = 1
    icount = 2

    def rand(self, shp):
        return numpy.random.rand(*shp).astype('float32')

    def setUp(self):
        numpy.random.seed(234)

        fModulesR = (self.imshp[0] - self.ksize + 1) // self.subsample_stride
        fModulesC = fModulesR
        self.fshape = (fModulesR, fModulesC, self.channels // self.ngroups,
                self.ksize, self.ksize, self.ngroups, self.nkern_per_group)
        self.ishape = (self.ngroups, self.channels // self.ngroups,
                self.imshp[0], self.imshp[1], self.icount)
        self.hshape = (self.ngroups, self.nkern_per_group, fModulesR, fModulesC,
                self.icount)

        filters = theano.shared(self.rand(self.fshape))

        self.A = LocalDot(filters, self.imshp[0], self.imshp[1],
                subsample=(self.subsample_stride, self.subsample_stride))

        self.xlval = self.rand((self.hshape[-1],) + self.hshape[:-1])
        self.xrval = self.rand(self.ishape)

        self.xl = theano.shared(self.xlval)
        self.xr = theano.shared(self.xrval)


