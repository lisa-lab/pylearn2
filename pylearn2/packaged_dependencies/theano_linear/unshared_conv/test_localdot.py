from __future__ import print_function

import nose
import unittest

import numpy as np
from theano.compat.six.moves import xrange
import theano

from .localdot import LocalDot

from ..test_matrixmul import SymbolicSelfTestMixin


class TestLocalDot32x32(unittest.TestCase, SymbolicSelfTestMixin):
    channels = 3
    bsize = 10     # batch size
    imshp = (32, 32)
    ksize = 5
    nkern_per_group = 16
    subsample_stride = 1
    ngroups = 1

    def rand(self, shp):
        return np.random.rand(*shp).astype('float32')

    def setUp(self):
        np.random.seed(234)
        assert self.imshp[0] == self.imshp[1]
        fModulesR = (self.imshp[0] - self.ksize + 1) // self.subsample_stride
        #fModulesR += 1 # XXX GpuImgActs crashes w/o this??
        fModulesC = fModulesR
        self.fshape = (fModulesR, fModulesC, self.channels // self.ngroups,
                self.ksize, self.ksize, self.ngroups, self.nkern_per_group)
        self.ishape = (self.ngroups, self.channels // self.ngroups,
                self.imshp[0], self.imshp[1], self.bsize)
        self.hshape = (self.ngroups, self.nkern_per_group, fModulesR, fModulesC,
                self.bsize)

        filters = theano.shared(self.rand(self.fshape))

        self.A = LocalDot(filters, self.imshp[0], self.imshp[1],
                subsample=(self.subsample_stride, self.subsample_stride))

        self.xlval = self.rand((self.hshape[-1],) + self.hshape[:-1])
        self.xrval = self.rand(self.ishape)

        self.xl = theano.shared(self.xlval)
        self.xr = theano.shared(self.xrval)

    # N.B. the tests themselves come from SymbolicSelfTestMixin


class TestLocalDotLargeGray(TestLocalDot32x32):

    channels = 1
    bsize = 128
    imshp = (256, 256)
    ksize = 9
    nkern_per_group = 16
    subsample_stride = 2
    ngroups = 1
    n_patches = 3000

    def rand(self, shp):
        return np.random.rand(*shp).astype('float32')

    # not really a test, but important code to support
    # Currently exposes error, by e.g.:
    #  CUDA_LAUNCH_BLOCKING=1
    #  THEANO_FLAGS=device=gpu,mode=DEBUG_MODE
    #  nosetests -sd test_localdot.py:TestLocalDotLargeGray.run_autoencoder
    def run_autoencoder(
        self,
        n_train_iter=10000,   # -- make this small to be a good unit test
        rf_shape=(9, 9),
        n_filters=1024,
        dtype='float32',
        module_stride=2,
        lr=0.01,
        show_filters=True,
        ):
        if show_filters:
            # import here to fail right away
            import matplotlib.pyplot as plt

        try:
            import skdata.vanhateren.dataset
        except ImportError:
            raise nose.SkipTest()

        # 1. Get a set of image patches from the van Hateren data set
        print('Loading van Hateren images')
        n_images = 50
        vh = skdata.vanhateren.dataset.Calibrated(n_images)
        patches = vh.raw_patches((self.n_patches,) + self.imshp,
                                 items=vh.meta[:n_images],
                                 rng=np.random.RandomState(123),
                                )
        patches = patches.astype('float32')
        patches /= patches.reshape(self.n_patches, self.imshp[0] * self.imshp[1])\
            .max(axis=1)[:, None, None]
        # TODO: better local contrast normalization

        if 0 and show_filters:
            plt.subplot(2, 2, 1); plt.imshow(patches[0], cmap='gray')
            plt.subplot(2, 2, 2); plt.imshow(patches[1], cmap='gray')
            plt.subplot(2, 2, 3); plt.imshow(patches[2], cmap='gray')
            plt.subplot(2, 2, 4); plt.imshow(patches[3], cmap='gray')
            plt.show()

        # -- Convert patches to localdot format:
        #    groups x colors x rows x cols x images
        patches5 = patches[:, :, :, None, None].transpose(3, 4, 1, 2, 0)
        print('Patches shape', patches.shape, self.n_patches, patches5.shape)

        # 2. Set up an autoencoder
        print('Setting up autoencoder')
        hid = theano.tensor.tanh(self.A.rmul(self.xl))
        out = self.A.rmul_T(hid)
        cost = ((out - self.xl) ** 2).sum()
        params = self.A.params()
        gparams = theano.tensor.grad(cost, params)
        train_updates = [(p, p - lr / self.bsize * gp)
                         for (p, gp) in zip(params, gparams)]
        if 1:
            train_fn = theano.function([], [cost], updates=train_updates)
        else:
            train_fn = theano.function([], [], updates=train_updates)

        theano.printing.debugprint(train_fn)

        # 3. Train it
        params[0].set_value(0.001 * params[0].get_value())
        for ii in xrange(0, self.n_patches, self.bsize):
            self.xl.set_value(patches5[:, :, :, :, ii:ii + self.bsize], borrow=True)
            cost_ii, = train_fn()
            print('Cost', ii, cost_ii)

        if 0 and show_filters:
            self.A.imshow_gray()
            plt.show()

        assert cost_ii < 0  # TODO: determine a threshold for detecting regression bugs


