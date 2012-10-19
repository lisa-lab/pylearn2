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
from theano.tensor.nnet.conv import conv2d
from theano import function
import time
import matplotlib.pyplot as plt


def make_funcs(batch_size, rows, cols, channels, filter_rows,
        num_filters):
    rng = np.random.RandomState([2012,10,9])

    filter_cols = filter_rows

    base_image_value = rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32')
    base_filters_value = rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32')
    images = shared(base_image_value)
    filters = shared(base_filters_value, name='filters')

    # bench.py should always be run in gpu mode so we should not need a gpu_from_host here
    layer_1_detector = FilterActs()(images, filters)

    layer_1_pooled_fake = layer_1_detector[:,0:layer_1_detector.shape[0]:2,
            0:layer_1_detector.shape[1]:2, :]

    base_filters2_value = rng.uniform(-1., 1., (num_filters, filter_rows,
        filter_cols, num_filters)).astype('float32')
    filters2 = shared(base_filters_value, name='filters')

    layer_2_detector = FilterActs()(images, filters2)

    output = layer_2_detector

    output_shared = shared( output.eval() )

    cuda_convnet = function([], updates = { output_shared : output } )
    cuda_convnet.name = 'cuda_convnet'

    images_bc01 = base_image_value.transpose(3,0,1,2)
    filters_bc01 = base_filters_value.transpose(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    images_bc01 = shared(images_bc01)
    filters_bc01 = shared(filters_bc01)

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    output_conv2d_shared = shared(output_conv2d.eval())

    baseline = function([], updates = { output_conv2d_shared : output_conv2d } )
    baseline.name = 'baseline'

    return cuda_convnet, baseline

def bench(f):
    for i in xrange(3):
        f()
    trials = 10
    t1 = time.time()
    for i in xrange(trials):
        f()
    t2 = time.time()
    return (t2-t1)/float(trials)

def get_speedup( *args, **kwargs):
    cuda_convnet, baseline = make_funcs(*args, **kwargs)
    return bench(baseline) / bench(cuda_convnet)

def get_time_per_10k_ex( *args, **kwargs):
    cuda_convnet, baseline = make_funcs(*args, **kwargs)
    batch_size = kwargs['batch_size']
    return 10000 * bench(cuda_convnet) / float(batch_size)

def make_batch_size_plot(yfunc, yname, batch_sizes, rows, cols, channels, filter_rows, num_filters):
    speedups = []
    for batch_size in batch_sizes:
        speedup = yfunc(batch_size = batch_size,
                rows = rows,
                cols = cols,
                channels = channels,
                filter_rows = filter_rows,
                num_filters = num_filters)
        speedups.append(speedup)
    plt.plot(batch_sizes, speedups)
    plt.title("cuda-convnet benchmark")
    plt.xlabel("Batch size")
    plt.ylabel(yname)
    plt.show()

"""
make_batch_size_plot(get_speedup, "Speedup factor", batch_sizes = [1,2,5,25,32,50,63,64,65,96,100,127,128,129,159,160,161,191,192,193,200,255,256,257],
        rows = 32,
        cols = 32,
        channels = 3,
        filter_rows = 7,
        num_filters = 64)
"""

make_batch_size_plot(get_time_per_10k_ex, "Time per 10k examples", batch_sizes = [1,2,5,25,32,50,63,64,65,96,100,127,128,129,159,160,161,191,192,193,200,255,256,257],
        rows = 32,
        cols = 32,
        channels = 3,
        filter_rows = 5,
        num_filters = 64)

