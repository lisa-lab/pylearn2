from __future__ import print_function

import unittest
from nose.plugins.skip import SkipTest
import numpy

import theano

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
        raise SkipTest('Optional package cuda disabled')

from theano.sandbox.cuda.var import float32_shared_constructor

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

from .gpu_unshared_conv import (
        GpuFilterActs,
        GpuWeightActs,
        GpuImgActs,
        )

import test_unshared_conv


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestGpuFilterActs(test_unshared_conv.TestFilterActs):
    """
    This class tests GpuWeightActs via the gradient of GpuFilterAct

    The correctness of GpuFilterActs is tested in TestMatchFilterActs
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1
    dtype = 'float32'
    mode = theano.compile.get_default_mode().including('gpu_opt',
            'fast_run', 'inplace').including('gpu_after_fusion',
                    'fast_run', 'inplace')

    def setUp(self):
        test_unshared_conv.TestFilterActs.setUp(self)
        self.gpu_op = GpuFilterActs(
                module_stride=self.module_stride,
                partial_sum=1)
        self.s_images = float32_shared_constructor(
                self.s_images.get_value())
        self.s_filters = float32_shared_constructor(
                self.s_filters.get_value())

    def test_gpu_shape(self):
        import theano.sandbox.cuda as cuda_ndarray
        if cuda_ndarray.cuda_available == False:
            raise SkipTest('Optional package cuda disabled')
        gpuout = self.gpu_op(self.s_images, self.s_filters)
        assert 'Cuda' in str(self.s_filters.type)
        f = theano.function([], gpuout, mode=mode_with_gpu)
        outval = f()
        assert outval.shape == (
                self.fshape[-2], self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_insert_gpu_filter_acts(self):
        out = self.op(self.s_images, self.s_filters)
        f = self.function([], out)
        try:
            fgraph = f.maker.fgraph
        except:
            # this needs to work for older versions of theano too
            fgraph = f.maker.env
        assert isinstance(
                fgraph.toposort()[0].op,
                GpuFilterActs)

    def test_gpu_op_eq(self):
        assert GpuFilterActs(1, 1) == GpuFilterActs(1, 1)
        assert not (GpuFilterActs(1, 1) != GpuFilterActs(1, 1))
        assert (GpuFilterActs(1, 2) != GpuFilterActs(1, 1))
        assert (GpuFilterActs(2, 1) != GpuFilterActs(1, 1))
        assert GpuFilterActs(2, 1) != None

class TestGpuWeightActs(unittest.TestCase):
    """
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    hshape = (1, 16, 2, 2, 2)
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    frows = 3
    fcols = 3
    module_stride = 1
    partial_sum = 1
    dtype = 'float32'

    def setUp(self):
        self.gwa = GpuWeightActs(
                module_stride=self.module_stride,
                partial_sum=self.partial_sum)
        self.gpu_images = float32_shared_constructor(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.gpu_hidact = float32_shared_constructor(
                numpy.random.rand(*self.hshape).astype(self.dtype))

    def test_shape(self):
        dfilters = self.gwa(self.gpu_images, self.gpu_hidact,
                self.frows, self.fcols)
        f = theano.function([], dfilters)
        outval = f()
        assert outval.shape == self.fshape

class TestGpuImgActs(unittest.TestCase):
    """
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    hshape = (1, 16, 2, 2, 2)
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    irows = 4
    icols = 4
    module_stride = 1
    partial_sum = 1
    dtype = 'float32'

    def setUp(self):
        self.gia = GpuImgActs(
                module_stride=self.module_stride,
                partial_sum=self.partial_sum)
        self.gpu_images = float32_shared_constructor(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.gpu_hidact = float32_shared_constructor(
                numpy.random.rand(*self.hshape).astype(self.dtype))
        self.gpu_filters = float32_shared_constructor(
                numpy.random.rand(*self.fshape).astype(self.dtype))

    def test_shape(self):
        dimages = self.gia(self.gpu_filters, self.gpu_hidact,
                self.irows, self.icols)
        f = theano.function([], dimages)
        outval = f()
        assert outval.shape == self.ishape


if 1:
  class TestMatchFilterActs(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(77)

    def run_match(self, images, filters, module_stride, retvals=False, partial_sum=1):

        gfa = GpuFilterActs(module_stride, partial_sum)
        fa = FilterActs(module_stride)

        gpu_images = float32_shared_constructor(images)
        gpu_filters = float32_shared_constructor(filters)
        cpu_images = theano.shared(images)
        cpu_filters = theano.shared(filters)

        gpu_out = gfa(gpu_images, gpu_filters)
        cpu_out = fa(cpu_images, cpu_filters)

        f = theano.function([], [cpu_out, gpu_out])
        cpuval, gpuval = f()
        gpuval = numpy.asarray(gpuval)

        if retvals:
            return cpuval, gpuval
        else:
            #print 'run_match: cpu shape', cpuval.shape
            #print 'run_match: gpu shape', gpuval.shape
            assert cpuval.shape == gpuval.shape
            assert numpy.allclose(cpuval, gpuval)

    def run_match_shape(self, ishape, fshape, module_stride, dtype='float32'):
        return self.run_match(
            images=numpy.random.rand(*ishape).astype(dtype),
            filters=numpy.random.rand(*fshape).astype(dtype),
            module_stride=module_stride)

    def test_small_random(self):
        self.run_match_shape(
            ishape = (1, 1, 4, 4, 2),
            fshape = (2, 2, 1, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_colors(self):
        self.run_match_shape(
            ishape = (1, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_groups(self):
        self.run_match_shape(
            ishape = (5, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 5, 16),
            module_stride = 1)

    def test_small_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 5, 5, 1),
            fshape = (2, 2, 6, 3, 3, 4, 16),
            module_stride = 2)

    def test_med_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 32, 32, 1),
            fshape = (12, 12, 6, 3, 3, 4, 16),
            module_stride = 2)


    def _blah_topcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[0,0,0,0,0,0,0] = 1
        self.run_match(images, filters, 1)

    def _blah_botcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[1,1,0,0,0,0,0] = 1
        cpuval, gpuval = self.run_match(images, filters, 1, retvals=True)
        print(images)
        print(cpuval[:, :, 1, 1, :])
        print(gpuval[:, :, 1, 1, :])

