__authors__ = "Jan Schlueter"
__credits__ = ["Jan Schlueter"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA lab"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano import shared
import itertools
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous, host_from_gpu
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano import function
import warnings

def check_results(a, b, tol):
    if np.abs(a - b).max() > tol:
        assert type(a) == type(b)
        assert a.dtype == b.dtype
        if a.shape != b.shape:
            print 'cuda-convnet shape: ',a.shape
            print 'theano shape: ',b.shape
            assert False
        err = np.abs(a - b)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (a.min(), a.max())
        print 'theano value range: ', (b.min(), b.max())
        assert False


def test_match_grouped_conv():

    # Tests that convolution with numGroups > 1 does the right thing.

    for partial_sum in [0, 1, 4]:
        for groups in [2, 4]:
            rng = np.random.RandomState([2014,5,19])

            # define inputs
            batch_size = 128
            rows = 7
            cols = 9
            channels = 16  # must be a multiple of groups*4 for the weight gradient to work
            filter_rows = 4
            filter_cols = filter_rows
            num_filters = 64  # must be a multiple of groups*16 for anything to work

            images = shared(rng.uniform(-1., 1., (channels, rows, cols,
                batch_size)).astype('float32'), name='images')
            filters = shared(rng.uniform(-1., 1., (channels / groups, filter_rows,
                filter_cols, num_filters)).astype('float32'), name='filters')

            # define graph using cuda-convnet
            gpu_images = gpu_contiguous(images)
            gpu_filters = gpu_contiguous(filters)

            output = FilterActs(partial_sum=partial_sum, groups=groups)(gpu_images, gpu_filters)
            image_grad = T.grad(output.sum(), images)
            filter_grad = T.grad(output.sum(), filters)

            # define graph using theano's conv2d
            # that's a bit cumbersome because we need to handle groups ourselves
            # - convert from c01b to bc01 layout and flip filters
            images_bc01 = images.dimshuffle(3,0,1,2)
            filters_bc01 = filters.dimshuffle(3,0,1,2)[:,:,::-1,::-1]
            # - split images by channels
            channels_per_group = channels / groups
            split_images = [images_bc01[:, c*channels_per_group:(c+1)*channels_per_group]
                    for c in xrange(groups)]
            # - split filters
            filters_per_group = num_filters / groups
            split_filters = [filters_bc01[c*filters_per_group:(c+1)*filters_per_group]
                    for c in xrange(groups)]
            # - convolve each group
            split_outputs = [conv2d(input=inp, filters=filt,
                    image_shape=(batch_size, channels_per_group, rows, cols),
                    filter_shape=(filters_per_group, channels_per_group, filter_rows, filter_cols))
                    for inp, filt in itertools.izip(split_images, split_filters)]
            # - join output channels
            output_conv2d = T.join(1, *split_outputs)
            # - convert back from bc01 layout to c01b layout
            output_conv2d = output_conv2d.dimshuffle(1,2,3,0)
            # - define gradients (in c01b layout right away)
            image_grad_conv2d = T.grad(output_conv2d.sum(), images)
            filter_grad_conv2d = T.grad(output_conv2d.sum(), filters)

            # compile and call function
            f = function([], [host_from_gpu(output), output_conv2d, host_from_gpu(image_grad), host_from_gpu(image_grad_conv2d), host_from_gpu(filter_grad), host_from_gpu(filter_grad_conv2d)])
            output, output_conv2d, image_grad, image_grad_conv2d, filter_grad, filter_grad_conv2d = f()

            # check results
            check_results(output, output_conv2d, 8e-6)
            check_results(image_grad, image_grad_conv2d, 2e-5)
            check_results(filter_grad, filter_grad_conv2d, 3e-4)

            warnings.warn("""test_grouped_conv success criterion is not very strict. Can we verify that this is OK?
                             One possibility is that theano is numerically unstable and Alex's code is better, or
                             vice versa, or rounding errors just propagate differently.
                             Probably theano CPU 64 bit is OK but it's worth checking the others.""")

if __name__ == '__main__':
    test_match_grouped_conv()

