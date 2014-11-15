"""

This file implements feature extraction using the k-means + triangle
code with random pooling from the Jia and Huang paper.

Usage: python feature_extractor.py config.yaml

config.yaml should deserialize to an extract_features.FeatureExtractor object.
See the FeatureExtractor.__init__ method for documentation of options.

"""


from __future__ import print_function

import os
from pylearn2.config import yaml_parse
import warnings
import time
import copy
import numpy as np
from theano.compat.six.moves import xrange
from theano import config
from theano import tensor as T
from theano import function
from pylearn2.datasets.preprocessing import ExtractPatches, ExtractGridPatches, ReassembleGridPatches
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.datasets.tl_challenge import TL_Challenge
from pylearn2.expr.coding import triangle_code
from pylearn2.space import VectorSpace


import sys
config.floatX = 'float32'

#This is hard-coded to the value from the Jia and Huang paper
#I confirmed via e-mail with the authors that they indeed create
#4 superpixels along each axis, not superpixels of size 4, and
#that not all superpixels are the same size (since
#the dimensions of the detector feature map are not divisible by 4)
num_superpixels = 4

class halver:
    """
    This object splits the processing for an example in half.
    If you're using a lot of features, this is necessary to be able
    to fit the computation on the GPU

    Other than that, you can just pretend it's a theano function
    """

    def __init__(self,f,nhid):
        """ f: the theano function whose work needs to be split
            nhid: we assume f takes a matrix of shape (m,nvis) and
                  returns a matrix of shape (m,nhid)
        """
        self.f = f
        self.nhid = nhid

    def __call__(self,X):
        m = X.shape[0]
        #mid = m/2
        m1 = m/3
        m2 = 2*m/3

        rval = np.zeros((m,self.nhid),dtype='float32')
        rval[0:m1,:] = self.f(X[0:m1,:])
        rval[m1:m2,:] = self.f(X[m1:m2,:])
        rval[m2:,:] = self.f(X[m2:,:])

        return rval


class FeaturesDataset:
    def __init__(self, dataset_maker, num_examples, pipeline_path):
        """
            dataset_maker: A callable that returns a Dataset
            num_examples: the number of examples we expect the dataset to have
                          (just for error checking purposes)
        """
        self.dataset_maker = dataset_maker
        self.num_examples = num_examples
        self.pipeline_path = pipeline_path


stl10 = {}
stl10_no_shelling = {}
stl10_size = { 'train' : 5000, 'test' : 8000 }

cifar10 = {}
cifar10_size = { 'train' : 50000, 'test' : 10000 }

cifar100 = {}
cifar100_size = { 'train' : 50000, 'test' : 10000 }

tl_challenge = {}
tl_challenge_size = { 'train' : 120, 'test' : 0 }

class dataset_loader:
    def __init__(self, path):
        self.path = path

    def __call__(self):
        return serial.load(self.path)

class dataset_constructor:
    def __init__(self, cls, which_set):
        self.cls = cls
        self.which_set = which_set

    def __call__(self):
        return self.cls(self.which_set)


for which_set in ['train', 'test']:
    stl10[which_set] = {}
    stl10_no_shelling[which_set] = {}
    cifar10[which_set] = {}
    cifar100[which_set] = {}
    tl_challenge[which_set] = {}

    #this is for patch size, not datset size
    for size in [6]:
        stl10[which_set][size] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${PYLEARN2_DATA_PATH}/stl10/stl10_patches/preprocessor.pkl')

        stl10_no_shelling[which_set][size] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${GOODFELI_TMP}/stl10/stl10_patches_no_shelling/preprocessor.pkl')

        cifar10[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( CIFAR10, which_set),
                                                num_examples = cifar10_size[which_set],
                                                pipeline_path = '${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M_6x6.pkl')

        cifar100[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( CIFAR100, which_set),
                                                num_examples = cifar100_size[which_set],
                                                pipeline_path = '${PYLEARN2_DATA_PATH}/cifar100/cifar100_patches/preprocessor.pkl')

        tl_challenge[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( TL_Challenge, which_set),
                                                num_examples = tl_challenge_size[which_set],
                                                pipeline_path = '${GOODFELI_TMP}/tl_challenge_patches_2M_6x6_prepro.pkl')


    stl10[which_set][8] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${PYLEARN2_DATA_PATH}/stl10/stl10_patches_8x8/preprocessor.pkl')



class FeatureExtractor:
    def __init__(self, batch_size, kmeans_path,
           save_path,  dataset_family, which_set,
           num_output_features,
           chunk_size = None, restrict = None, pool_mode = 'mean'):
        """
            batch_size:  the number of images to process simultaneously
                         this does not affect the final result, it is just for performance
                         larger values allow more parallel processing but require more memory
            kmeans_path: a path to a .pkl file containing a pylearn2.kmeans.KMeans instance
            save_path:   the base path to save to, should end in .npy
            dataset_family: extract_features.stl10, extract_features.cifar10, etc.
            which_set:     'train' or 'test'
            num_output_features: the number of randomly selected pooled features to extract per image
            chunk_size:   will build a design matrix of this many processed examples before serializing
                         them. if you use a chunk size of 10,000 on a dataset with 50,000 examples, and
                         a save_path of foo.npy, this will result in you obtaining foo_A.npy through
                         foo_E.npy.
                         Use a small chunk_size to avoid running out of memory.
            restrict:    a tuple of of (start,end) indices
                         restrict feature extraction to only these examples
                         the restrict option is used internally to implement the chunk_size option, so
                         you may not specify both for the same FeatureExtractor
            pool_mode:   'max' or 'mean'
        """

        if chunk_size is not None and restrict is not None:
            raise NotImplementedError("Currently restrict is used internally to "
                    "implement chunk_size, so a client may not specify both")

        self.batch_size = batch_size
        self.model_path = kmeans_path
        self.restrict = restrict
        self.pool_mode = pool_mode


        assert save_path is not None
        assert save_path.endswith('npy')

        assert pool_mode in [ 'mean', 'max' ]

        self.save_path = save_path
        self.which_set = which_set
        self.dataset_family = dataset_family
        self.chunk_size = chunk_size
        self.num_output_features = num_output_features


    def __call__(self):

        print('loading model')
        model_path = self.model_path
        self.model = serial.load(model_path)
        self.model.set_dtype('float32')
        input_space = self.model.get_input_space()
        #This code all assumes the model works on vectors
        assert isinstance(input_space, VectorSpace)
        nvis = input_space.dim
        self.size = int(np.sqrt(nvis/3))

        rng = make_np_rng(None, [1,2,3], which_method="randint")

        #Generate the random pooling structure
        num_filters = self.model.mu.get_value().shape[0]

        idxs = rng.randint(0,num_filters,(self.num_output_features,))
        top = idxs.copy()
        bottom = idxs.copy()
        left = idxs.copy()
        right = idxs.copy()
        for i in xrange(self.num_output_features):
            top[i] = rng.randint(num_superpixels)
            bottom[i] = rng.randint(top[i],num_superpixels)
            left[i] = rng.randint(num_superpixels)
            right[i] = rng.randint(left[i],num_superpixels)

        #output[i] will pool over detector feature map i
        self.idxs = idxs
        #output [i] will pool over a rectangle with upper left coordinates (top[i],left[i])
        #and lower right coordinates (bottom[i],right[i])
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right



        #Run the experiment
        if self.chunk_size is not None:
            dataset_family = self.dataset_family
            which_set = self.which_set
            dataset_descriptor = self.dataset_family[which_set][size]

            num_examples = dataset_descriptor.num_examples
            assert num_examples % self.chunk_size == 0

            self.chunk_id = 0
            for i in xrange(0,num_examples, self.chunk_size):
                self.restrict = (i, i + self.chunk_size)

                self._execute()

                self.chunk_id += 1
        else:
            self._execute()

    def _execute(self):

        global num_superpixels
        num_output_features = self.num_output_features
        idxs = self.idxs
        top = self.top
        bottom = self.bottom
        left = self.left
        right = self.right

        save_path = self.save_path
        batch_size = self.batch_size
        dataset_family = self.dataset_family
        which_set = self.which_set
        model = self.model
        size = self.size

        nan = 0


        dataset_descriptor = dataset_family[which_set][size]

        dataset = dataset_descriptor.dataset_maker()
        expected_num_examples = dataset_descriptor.num_examples

        full_X = dataset.get_design_matrix()
        num_examples = full_X.shape[0]
        assert num_examples == expected_num_examples

        if self.restrict is not None:
            assert self.restrict[1]  <= full_X.shape[0]

            print('restricting to examples ',self.restrict[0],' through ',self.restrict[1],' exclusive')
            full_X = full_X[self.restrict[0]:self.restrict[1],:]

            assert self.restrict[1] > self.restrict[0]

        #update for after restriction
        num_examples = full_X.shape[0]

        assert num_examples > 0

        dataset.X = None
        dataset.design_loc = None
        dataset.compress = False

        patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )

        pipeline = serial.load(dataset_descriptor.pipeline_path)

        assert isinstance(pipeline.items[0], ExtractPatches)
        pipeline.items[0] = patchifier


        print('defining features')
        V = T.matrix('V')

        mu = model.mu

        feat = triangle_code(V, mu)


        assert feat.dtype == 'float32'
        print('compiling theano function')
        f = function([V],feat)

        nhid = model.mu.get_value().shape[0]

        if config.device.startswith('gpu') and nhid >= 4000:
            f = halver(f, model.nhid)

        topo_feat_var = T.TensorType(broadcastable = (False,False,False,False), dtype='float32')()
        if self.pool_mode == 'mean':
            region_features = function([topo_feat_var],
                topo_feat_var.mean(axis=(1,2)) )
        elif self.pool_mode == 'max':
            region_features = function([topo_feat_var],
                    topo_feat_var.max(axis=(1,2)) )
        else:
            assert False

        def average_pool( stride ):
            def point( p ):
                return p * ns / stride

            rval = np.zeros( (topo_feat.shape[0], stride, stride, topo_feat.shape[3] ) , dtype = 'float32')

            for i in xrange(stride):
                for j in xrange(stride):
                    rval[:,i,j,:] = region_features( topo_feat[:,point(i):point(i+1), point(j):point(j+1),:] )

            return rval

        output =  np.zeros((num_examples,num_output_features),dtype='float32')


        fd = DenseDesignMatrix(X = np.zeros((1,1),dtype='float32'), view_converter = DefaultViewConverter([1, 1, nhid] ) )

        ns = 32 - size + 1
        depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )

        if len(range(0,num_examples-batch_size+1,batch_size)) <= 0:
            print(num_examples)
            print(batch_size)

        for i in xrange(0,num_examples-batch_size+1,batch_size):
            print(i)
            t1 = time.time()

            d = copy.copy(dataset)
            d.set_design_matrix(full_X[i:i+batch_size,:])

            t2 = time.time()

            #print '\tapplying preprocessor'
            d.apply_preprocessor(pipeline, can_fit = False)
            X2 = d.get_design_matrix()

            t3 = time.time()

            #print '\trunning theano function'
            feat = f(X2)

            t4 = time.time()

            assert feat.dtype == 'float32'

            feat_dataset = copy.copy(fd)

            if contains_nan(feat):
                nan += np.isnan(feat).sum()
                feat[np.isnan(feat)] = 0

            feat_dataset.set_design_matrix(feat)

            #print '\treassembling features'
            feat_dataset.apply_preprocessor(depatchifier)

            #print '\tmaking topological view'
            topo_feat = feat_dataset.get_topological_view()
            assert topo_feat.shape[0] == batch_size

            t5 = time.time()

            #average pooling
            superpixels = average_pool(num_superpixels)

            assert batch_size == 1

            if self.pool_mode == 'mean':
                for j in xrange(num_output_features):
                    output[i:i+batch_size, j] = superpixels[:,top[j]:bottom[j]+1,
                            left[j]:right[j]+1, idxs[j]].mean()
            elif self.pool_mode == 'max':
                for j in xrange(num_output_features):
                    output[i:i+batch_size, j] = superpixels[:,top[j]:bottom[j]+1,
                            left[j]:right[j]+1, idxs[j]].max()
            else:
                assert False

            assert output[i:i+batch_size,:].max() < 1e20

            t6 = time.time()

            print((t6-t1, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

        if self.chunk_size is not None:
            assert save_path.endswith('.npy')
            save_path_pieces = save_path.split('.npy')
            assert len(save_path_pieces) == 2
            assert save_path_pieces[1] == ''
            save_path = save_path_pieces[0] + '_' + chr(ord('A')+self.chunk_id)+'.npy'
        np.save(save_path,output)


        if nan > 0:
            warnings.warn(str(nan)+' features were nan')

if __name__ == '__main__':
    assert len(sys.argv) == 2
    yaml_path = sys.argv[1]

    assert yaml_path.endswith('.yaml')
    val = yaml_path[0:-len('.yaml')]
    os.environ['FEATURE_EXTRACTOR_YAML_PATH'] = val
    os.putenv('FEATURE_EXTRACTOR_YAML_PATH',val)
    val = val.split('/')[-1]
    os.environ['FEATURE_EXTRACTOR_YAML_NAME'] = val
    os.putenv('FEATURE_EXTRACTOR_YAML_NAME', val)


    extractor = yaml_parse.load_path(yaml_path)

    extractor()
