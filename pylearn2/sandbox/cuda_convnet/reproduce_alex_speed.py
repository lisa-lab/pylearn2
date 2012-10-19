__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
from theano import shared
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano import function
import time
import matplotlib.pyplot as plt
import warnings
from theano.printing import Print

# Alex can do 10k examples in < 2 sec
# His convolution kernels are ~2-5X faster than ours for batch sizes > 10 or so
# So we should be able to do 10k examples in 10 sec using theano
# (I don't want to use his kernels yet because I don't have them set up to work
# with gradient)


def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only support pool_stride = pool_shape
    so I have to make my own max pooling function to reproduce
    Alex's architecture.

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    last_valid_r = r - pr
    last_start_r = last_valid_r - last_valid_r % rs

    last_valid_c = c - pc
    last_start_c = last_valid_c - last_valid_c % cs

    for row_start in xrange(pool_shape[0]):
        row_stop = last_start_r + row_start + 1
        for col_start in xrange(pool_shape[1]):
            col_stop = last_start_c + col_start + 1
            cur = bc01[:,:,row_start:row_stop:rs,col_start:col_stop:cs]
            cur.name = 'max_pool_cur_'+bc01.name+'_'+str(row_start)+'_'+str(col_start)
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = 'max_pool_mx_'+bc01.name+'_'+str(row_start)+'_'+str(col_start)
    return mx

def max_pool_hack(bc01, pool_shape, pool_stride, image_shape, min_outer_shape):
    """
    Theano's max pooling op only support pool_stride = pool_shape
    so I have to make my own max pooling function to reproduce
    Alex's architecture.

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    mr, mc = min_outer_shape

    last_start_r = mr - mr % rs
    required_r = last_start_r + pr

    last_start_c = mc - mc % cs
    required_c = last_start_c + pc


    hack = T.alloc(0., bc01.shape[0], bc01.shape[1], required_r, required_c)

    name = bc01.name
    bc01 = T.set_subtensor(hack[:,:, 0:r, 0:c], bc01)
    bc01.name = 'hack_' + name


    for row_start in xrange(pool_shape[0]):
        row_stop = last_start_r + row_start + 1
        for col_start in xrange(pool_shape[1]):
            col_stop = last_start_c + col_start + 1
            cur = bc01[:,:,row_start:row_stop:rs,col_start:col_stop:cs]
            cur.name = 'max_pool_cur_'+bc01.name+'_'+str(row_start)+'_'+str(col_start)
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = 'max_pool_mx_'+bc01.name+'_'+str(row_start)+'_'+str(col_start)
    return mx

def make_funcs(batch_size, stage = -1):
    rng = np.random.RandomState([2012,10,9])

    params = []

    pylearn2_tax = 1

    # Input image
    images_value = rng.uniform(-1., 1.,
            (batch_size, 3, 24, 24)).astype('float32')

    if pylearn2_tax:
        images_value = rng.uniform(-1., 1., (batch_size, 24, 24, 3)).astype('float32')

    images = shared(images_value, name = 'images')

    if pylearn2_tax:
        images = images.dimshuffle((0,3,1,2))

    # conv1
    filters_value = rng.uniform(-1., 1.,
            (64, 3, 5, 5)).astype('float32')
    filters = shared(filters_value, name='filters')
    params.append(filters)

    layer_1_conv_only = conv2d(images, filters, border_mode = 'valid', image_shape = (batch_size, 3, 24, 24), filter_shape = filters_value.shape)
    layer_1_conv_only.name = 'layer_1_conv_only'


    biases_value = rng.uniform(-1., 1., (64,)).astype('float32')
    biases = shared(biases_value, name = 'biases')
    params.append(biases)

    broadcasted_biases = biases.dimshuffle('x',0,'x','x')
    broadcasted_biases.name = 'broadcased_biases'
    layer_1_presynaptic = layer_1_conv_only + broadcasted_biases
    layer_1_presynaptic.name = 'layer_1_presynaptic'

    conv1 = layer_1_presynaptic * (layer_1_presynaptic > 0.)
    #conv1 = Print('conv1', attrs=['shape'])(conv1)
    conv1.name = 'conv1'
    # shape is (batch_size, 64, 20, 20)

    # pool1
    # make image 4 pixels wider to fake it being made with pad=2
    pool1 = max_pool_hack(conv1, pool_shape = (3, 3), pool_stride = (2, 2), image_shape = (20,20), min_outer_shape = (24, 24))
    #pool1 = Print('pool1',attrs=['shape'])(pool1)
    pool1.name = 'pool1'
    # shape is batch_size, 64, 13, 13

    # conv2
    filters2_value = rng.uniform(-1., 1., (64, 64, 5, 5)).astype('float32')
    filters2 = shared(filters2_value, name='filters2')
    params.append(filters2)
    layer_2_conv_only = conv2d(pool1, filters2, border_mode = 'valid', image_shape = (batch_size, 64, 13, 13),
            filter_shape = filters2_value.shape)
    layer_2_conv_only.name = 'layer_2_conv_only'
    biases2_value = rng.uniform(-1., 1., (64,)).astype('float32')
    biases2 = shared(biases2_value, name='biases2')
    params.append(biases2)
    broadcasted_biases2 = biases2.dimshuffle('x',0,'x','x')
    layer_2_presynaptic = layer_2_conv_only + broadcasted_biases2
    layer_2_presynaptic.name = 'layer_2_presynaptic'
    conv2 = layer_2_presynaptic * (layer_2_presynaptic > 0.)
    #conv2 = Print('conv2', attrs=['shape'])(conv2)
    conv2.name = 'conv2'
    # conv2 has shape (batch_size, 64, 9, 9)

    # pool2
    # make image 4 pixels wider to fake it being made with pad=2
    pool2 = max_pool_hack(conv2, pool_shape=(3, 3), pool_stride=(2, 2), image_shape=(9, 9), min_outer_shape = (13, 13))
    #pool2 = Print('pool2', attrs=['shape'])(pool2)
    pool2.name = 'pool2'
    # shape 7x7

    # local3
    warnings.warn("theano doesn't have a local rf operator, so we just use convolution for local3")
    warnings.warn("local3 uses pad=1")
    filters3_value = rng.uniform(-1., 1., (32, 64, 3, 3)).astype('float32')
    filters3 = shared(filters3_value, 'filters3')
    layer_3_conv_only = conv2d(pool2, filters3, border_mode='valid', image_shape=(batch_size, 64, 7, 7),
            filter_shape=filters3_value.shape)
    layer_3_conv_only.name = 'layer_3_conv_only'
    # This should be of shape 5,5
    biases3_value = rng.uniform(-1., 1., (32,5,5)).astype('float32')
    biases3 = shared(biases3_value, 'biases3')
    biases3_broadcasted = biases3.dimshuffle('x',0,1,2)
    layer_3_presynaptic = layer_3_conv_only + biases3_broadcasted
    layer_3_presynaptic.name = 'layer_3_presynaptic'
    local3 = layer_3_presynaptic * (layer_3_presynaptic > 0.)
    local3.name = 'local3'
    warnings.warn("local3 should have been computed with pad=1, we just ignore that here")

    #local4
    filters4_value = rng.uniform(-1., 1., (32, 32, 3, 3)).astype('float32')
    filters4 = shared(filters4_value, 'filters4')
    layer_4_conv_only = conv2d(local3, filters4, border_mode='valid', image_shape=(batch_size, 32, 5, 5),
            filter_shape=filters4_value.shape)
    layer_4_conv_only.name = 'layer_4_conv_only'



    stages = [layer_1_conv_only, layer_1_presynaptic, conv1, pool1, layer_2_conv_only, layer_2_presynaptic,
            conv2, pool2, layer_3_conv_only, layer_3_conv_only, local3, layer_4_conv_only]
    output = stages[stage]

    obj = output.mean()
    obj.name = 'obj'

    learning_rate = 1e-3
    momentum = .9

    updates = {}
    for param in params:
        inc = shared(param.get_value() * 0.)
        new_inc = momentum * inc - learning_rate * T.grad(obj, param, disconnected_inputs = 'ignore')
        update = param + new_inc
        #update = Print("update_"+param.name,attrs=['shape'])(update)
        updates[param] = update
        updates[inc] = new_inc


    cuda_convnet = function([], updates = updates )
    cuda_convnet.name = 'cuda_convnet'

    return cuda_convnet, output.name

def bench(f):
    print 'benchmarking...'
    for i in xrange(3):
        f()
    trials = 10
    t1 = time.time()
    for i in xrange(trials):
        f()
    t2 = time.time()
    print '...done'
    return (t2-t1)/float(trials)

def get_time_per_10k_ex( *args, **kwargs):
    cuda_convnet, name = make_funcs(*args, **kwargs)
    batch_size = kwargs['batch_size']
    return 10000 * bench(cuda_convnet) / float(batch_size), name

def make_batch_size_plot(yfunc, yname, batch_sizes):
    speedups = []
    for batch_size in batch_sizes:
        print 'batch size: ',batch_size
        speedup = yfunc(batch_size = batch_size)
        speedups.append(speedup)
    plt.plot(batch_sizes, speedups)
    plt.title("cuda-convnet benchmark")
    plt.xlabel("Batch size")
    plt.ylabel(yname)
    plt.show()

def make_stage_plot(batch_size):
    speedups = []
    stages = range(12)
    names = []
    for stage in stages:
        speedup, name = get_time_per_10k_ex(batch_size = batch_size, stage = stage)
        speedups.append(speedup)
        names.append(name)
    for stage, name, speedup in zip(stages, names, speedups):
        print 'stage',stage,':',name,':',speedup
    plt.plot(stages, speedups)
    plt.title("cuda-convnet benchmark")
    plt.hold(True)
    plt.plot(stages, [10]*len(stages))
    plt.xlabel("Stage")
    plt.ylabel("Runtime")
    plt.show()

#make_batch_size_plot(get_time_per_10k_ex, "Time per 10k examples", batch_sizes = [1,10,20,25,50,100,128])
make_stage_plot(128)
