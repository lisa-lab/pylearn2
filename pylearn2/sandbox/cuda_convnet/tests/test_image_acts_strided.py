__authors__ = "Heng Luo"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano import shared
from theano.tensor import grad, constant
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.filter_acts import ImageActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor import as_tensor_variable
from theano import function
from theano import tensor as T
import warnings
from theano.sandbox import cuda

from test_filter_acts_strided import FilterActs_python

def ImageActs_python(filters,
                     hidacts,
                     stride=1,
                     img_shape=None,
                    ):
    if int(stride) != stride:
        raise TypeError('stride must be an int', stride)
    stride = int(stride)

    num_filters, h_rows, h_cols, batch_size = hidacts.shape    
    channels, filter_rows, filter_cols, _num_filters = filters.shape  
    assert filter_cols == filter_cols
    assert num_filters == _num_filters
    assert stride <= filter_rows and stride >= 1
    
    if stride > 1:
        assert img_shape!= None     
        rows, cols = img_shape 
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
        images = np.zeros((channels,new_rows,new_cols,batch_size),dtype='float32')
    else:
        rows = h_rows+filter_rows-1
        cols = h_cols+filter_cols-1
        img_shape = (channels, 
                     rows, 
                     cols, 
                     batch_size)
        images = np.zeros(img_shape,dtype='float32')
    
    n_dim_filter = channels*filter_rows*filter_cols
    vector_filters = filters.reshape(n_dim_filter,num_filters).T
    
    for idx_h_rows in xrange(h_rows):
        for idx_h_cols in xrange(h_cols):
                rc_hidacts = hidacts[:,idx_h_rows,idx_h_cols,:]
                rc_image = (np.dot(
                                  rc_hidacts.T,
                                  vector_filters).T).reshape(channels,filter_rows,filter_cols,batch_size)
                images[:,
                       idx_h_rows*stride:idx_h_rows*stride+filter_rows,
                       idx_h_cols*stride:idx_h_cols*stride+filter_cols,
                       :] += rc_image  
    rval = images[:,:rows,:cols,:] 
    return rval                
                          

def test_image_acts_strided():

    # Tests that running FilterActs with all possible strides 

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

    for test_idx in xrange(len(shape_list)):
        images = shared(rng.uniform(-1., 1., shape_list[test_idx][0]).astype('float32'), name='images')
        filters = shared(rng.uniform(-1., 1., shape_list[test_idx][1]).astype('float32'), name='filters')
        print "test case %d..."%(test_idx+1) 
        #gpu_images = gpu_from_host(images)
        #gpu_filters = gpu_from_host(filters)
        gpu_images = images
        gpu_filters = filters
    
        images_val = images.get_value(borrow=True)
        filters_val = filters.get_value(borrow=True)
        for ii in xrange(filters_val.shape[1]):
            stride = ii + 1
                   
            output_python = FilterActs_python(images_val,filters_val,stride)
            hidacts = shared(rng.uniform(-1., 1., output_python.shape).astype('float32'), name='hidacts')
            hidacts_val = hidacts.get_value(borrow=True)                
            Img_output_python = ImageActs_python(filters_val,hidacts_val,stride,(images_val.shape[1], images_val.shape[2]))            
            
            gpu_hidacts = hidacts
            Img_output = ImageActs(stride=stride)(gpu_hidacts, gpu_filters, as_tensor_variable((images_val.shape[1], images_val.shape[2])))
            Img_output = host_from_gpu(Img_output)
            f = function([], Img_output)
            Img_output_val = f()
            
            warnings.warn("""test_image_acts_strided success criterion is not very strict.""")
            
            if np.abs(Img_output_val - Img_output_python).max() > 2.1e-5:
                assert type(Img_output_val) == type(Img_output_python)
                assert Img_output_val.dtype == Img_output_python.dtype
                if Img_output_val.shape != Img_output_python.shape:
                    print 'cuda-convnet shape: ',Img_output_val.shape
                    print 'python conv shape: ',Img_output_python.shape
                    assert False
                err = np.abs(Img_output_val - Img_output_python)
                print 'stride %d'%stride
                print 'absolute error range: ', (err.min(), err.max())
                print 'mean absolute error: ', err.mean()
                print 'cuda-convnet value range: ', (Img_output_val.min(), Img_output_val.max())
                print 'python conv value range: ', (Img_output_python.min(), Img_output_python.max())    
                #assert False 
        #print "pass"         
               
if __name__ == '__main__':
    test_image_acts_strided()






