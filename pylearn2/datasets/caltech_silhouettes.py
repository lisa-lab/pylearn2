import os
import numpy

import theano
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class CaltechSilhouettes(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, shuffle=False, one_hot = False,
                 dtype=theano.config.floatX):

        path = '${PYLEARN2_DATA_PATH}/caltech_silhouettes'
        x_fnames = {'train': 'train_data.npy',
                    'valid': 'val_data.npy',
                    'test' : 'test_data.npy'}
        assert which_set in x_fnames.keys()
        y_fnames = {'train': 'train_labels.npy',
                    'valid': 'val_labels.npy',
                    'test' : 'test_labels.npy'}
        assert which_set in y_fnames.keys()

        # we also expose the following details:
        self.img_shape = (28,28)
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 101

        # prepare loading
        x = serial.load(os.path.join(path, x_fnames[which_set]))
        y = serial.load(os.path.join(path, y_fnames[which_set]))
        X = numpy.cast[dtype](x)
        y = y[:,0] - 1

        self.one_hot = one_hot
        if one_hot:
            one_hot = numpy.zeros((y.shape[0],101), dtype=dtype)
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        if shuffle:
            self.shuffle_rng = numpy.random.RandomState([1,2,3])
            perm_idx = self.shuffle_rng.permutation(len(x))
            X = X[perm_idx, ...]
            y = y[perm_idx, ...]

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(CaltechSilhouettes,self).__init__(X = X, y = y, view_converter = view_converter)
        assert not numpy.any(numpy.isnan(self.X))
