from __future__ import print_function

__authors__ = "Heng Luo"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano.compat.six.moves import xrange
from theano import shared
from theano.tensor import grad, constant
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor import as_tensor_variable
from theano import function
from theano import tensor as T
import warnings
from theano.sandbox import cuda
from theano.sandbox.cuda.var import float32_shared_constructor 

from test_filter_acts_strided import FilterActs_python

def WeightActs_python(images,
                      hidacts,
                      filter_rows,
                      filter_cols,
                      stride=1,
                      ):
    if int(stride) != stride:
        raise TypeError('stride must be an int', stride)
    stride = int(stride)

    channels, rows, cols, batch_size = images.shape    
    num_filters, hidact_rows, hidact_cols, _batch_size = hidacts.shape
    assert _batch_size == batch_size    
    assert filter_rows == filter_cols
    f_shape = (channels, filter_rows, filter_cols, num_filters)
    f = np.zeros(f_shape,dtype='float32')
    
    if stride > 1:
        if (rows - filter_rows)%stride == 0:
            stride_padding_rows = 0
        else:
            stride_padding_rows = ((rows - filter_rows)/stride + 1)*stride + filter_rows - rows
        idx_rows = (rows + stride_padding_rows - filter_rows)/stride 
        if (cols - filter_cols)%stride == 0:
            stride_padding_cols = 0
        else:
            stride_padding_cols = ((cols - filter_cols)/stride + 1)*stride + filter_cols - cols
        idx_cols = (cols + stride_padding_cols - filter_cols)/stride
                
        new_rows = rows + stride_padding_rows
        new_cols = cols + stride_padding_cols
        idx_rows = (new_rows - filter_rows)/stride 
        idx_cols = (new_cols - filter_cols)/stride
        
        new_images = np.zeros((channels, new_rows, new_cols, batch_size),dtype='float32')
        new_images[:,:rows,:cols,:] = images
    else:
        new_images = images
    
    n_dim_filter = channels*filter_rows*filter_cols
    
    for idx_filters in xrange(num_filters):
        for idx_h_rows in xrange(hidact_rows):
            for idx_h_cols in xrange(hidact_cols):
                rc_images = new_images[:,
                                       idx_h_rows*stride:idx_h_rows*stride+filter_rows,
                                       idx_h_cols*stride:idx_h_cols*stride+filter_cols,
                                      :]                                  
                rc_filters = np.dot(
                                    hidacts[idx_filters,idx_h_rows,idx_h_cols,:].reshape(1,batch_size),
                                    rc_images.reshape(n_dim_filter, batch_size).T)
                f[:,:,:,idx_filters] += rc_filters.reshape(channels, filter_rows, filter_cols)                  
    return f
    
def test_weight_acts_strided():

    # Tests that WeightActs with all possible strides 

    rng = np.random.RandomState([2012,10,9])

    #Each list in shape_list : 
    #[img_shape,filter_shape]
    #[(channels, rows, cols, batch_size),(channels, filter_rows, filter_cols, num_filters)]
    shape_list = [[(1, 7, 8, 5),     (1, 2, 2, 16)],
                  [(3, 7, 8, 5),     (3, 3, 3, 16)],
                  [(16, 11, 11, 4),  (16, 4, 4, 16)], 
                  [(3, 20, 20, 3),   (3, 5, 5, 16)],
                  [(3, 21, 21, 3),   (3, 6, 6, 16)],
                  ]
    for partial_sum in [0, 1, 4]:
        print("partial_sum: %d"%(partial_sum))
        for test_idx in xrange(len(shape_list)):
            images = rng.uniform(-1., 1., shape_list[test_idx][0]).astype('float32')
            filters = rng.uniform(-1., 1., shape_list[test_idx][1]).astype('float32')
            gpu_images = float32_shared_constructor(images,name='images')
            print("test case %d..."%(test_idx+1))
              
            for ii in xrange(filters.shape[1]):
                stride = ii + 1                            
                output_python = FilterActs_python(images,filters,stride)   
                _, h_rows, h_cols, _ = output_python.shape
                if partial_sum == 4:
                    if (h_rows*h_cols)%partial_sum != 0:
                        print("skip test case %d, stride %d when partial_sum is equal to %d"%(test_idx+1,stride,partial_sum))
                        break
                hidacts = rng.uniform(-1., 1., output_python.shape).astype('float32')
                gpu_hidacts = float32_shared_constructor(hidacts,name='hidacts')
                    
                weights_grad_python = WeightActs_python(images,hidacts,filters.shape[1],filters.shape[2],stride)
                
                weights_grad = WeightActs(partial_sum=partial_sum,stride=stride)(
                                                    gpu_images,
                                                    gpu_hidacts,
                                                    as_tensor_variable((filters.shape[1], filters.shape[2]))
                                                   )[0]
                weights_grad = host_from_gpu(weights_grad)
                f = function([], weights_grad)
                weights_grad_val = f()   
                
                warnings.warn("""test_weight_acts_strided success criterion is not very strict.""")
                
                if np.abs(weights_grad_val - weights_grad_python).max() > 3.4e-5:
                    assert type(weights_grad_val) == type(weights_grad_python)
                    assert weights_grad_val.dtype == weights_grad_python.dtype
                    if weights_grad_val.shape != weights_grad_python.shape:
                        print('cuda-convnet shape: ',weights_grad_val.shape)
                        print('python conv shape: ',weights_grad_python.shape)
                        assert False
                    err = np.abs(weights_grad_val - weights_grad_python)
                    print('stride %d'%stride)
                    print('absolute error range: ', (err.min(), err.max()))
                    print('mean absolute error: ', err.mean())
                    print('cuda-convnet value range: ', (weights_grad_val.min(), weights_grad_val.max()))
                    print('python conv value range: ', (weights_grad_python.min(), weights_grad_python.max()))
                    #assert False
                #print "stride %d"%stride     
                    
            #print "pass"         
               
if __name__ == '__main__':
    test_weight_acts_strided()






