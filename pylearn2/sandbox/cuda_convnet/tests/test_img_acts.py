__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley", "Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano import function
import warnings

def test_match_full_conv():

    # Tests that running ImageActs with no padding is the same as running
    # theano's conv2D in valid mode

    rng = np.random.RandomState([2013, 1, 29])

    batch_size = 5
    rows = 10
    cols = 9
    channels = 3
    filter_rows = 4
    filter_cols = filter_rows
    num_filters = 16

    hid_acts = shared(rng.uniform(-1., 1., (num_filters,
                                            rows - filter_rows + 1,
                                            cols - filter_cols + 1,
                                            batch_size)
    ).astype('float32'), name='hidacts')

    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(hid_acts)
    gpu_filters = gpu_from_host(filters)

    output = ImageActs()(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    images_bc01 = hid_acts.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='full')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    f = function([], [output, output_conv2d])

    output, output_conv2d = f()

    warnings.warn("""test_match_full_conv success criterion is not very strict. Can we verify that this is OK?
                     One possibility is that theano is numerically unstable and Alex's code is better.
                     Probably theano CPU 64 bit is OK but it's worth checking the others.""")
    if np.abs(output - output_conv2d).max() > 2.4e-6:
        assert type(output) == type(output_conv2d)
        assert output.dtype == output_conv2d.dtype
        if output.shape != output_conv2d.shape:
            print 'cuda-convnet shape: ',output.shape
            print 'theano shape: ',output_conv2d.shape
            assert False
        err = np.abs(output - output_conv2d)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (output.min(), output.max())
        print 'theano value range: ', (output_conv2d.min(), output_conv2d.max())
        assert False

if __name__ == '__main__':
    test_match_full_conv()

