__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano import function
import warnings

def test_match_grad_valid_conv():

    # Tests that weightActs is the gradient of FilterActs
    # with respect to the weights.

    rng = np.random.RandomState([2012,10,9])

    batch_size = 3
    rows = 10
    cols = 9
    channels = 8
    filter_rows = 4
    filter_cols = filter_rows
    num_filters = 16

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    output = FilterActs()(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    images_bc01 = images.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    theano_rng = MRG_RandomStreams(2013 + 1 + 31)

    coeffs = theano_rng.normal(avg=0., std=1., size=output_conv2d.shape, dtype='float32')

    cost_conv2d = (coeffs * output_conv2d).sum()

    weights_grad_conv2d = T.grad(cost_conv2d, filters)

    cost = (coeffs * output).sum()
    hid_acts_grad = T.grad(cost, output)

    weights_grad = WeightActs()(gpu_images, gpu_from_host(hid_acts_grad))

    f = function([], [output, output_conv2d, weights_grad, weights_grad_conv2d])

    output, output_conv2d = f()

    warnings.warn("""test_match_valid_conv success criterion is not very strict. Can we verify that this is OK?
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
    test_match_grad_valid_conv()

