import numpy as N
from pylearn2.datasets import dense_design_matrix
import pylearn.datasets.MNIST as i_hate_python
from pylearn.datasets import icml07

class MNIST(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        #dear pylearn.datasets.MNIST: there is no such thing as the MNIST validation set. quit pretending that there is.
        orig = i_hate_python.train_valid_test(ntrain=60000,nvalid=0,ntest=10000)

        Xs = {
                'train' : orig.train.x,
                'test'  : orig.test.x
            }

        X = N.cast['float32'](Xs[which_set])

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #

#

class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False):

        orig = icml07.MNIST_rotated_background()

        Xs = {'train': orig.train.x,
              'valid': orig.valid.x,
              'test' : orig.test.x}
        X = N.cast['float32'](Xs[which_set])

        if center:
            X -= 0.5#X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST_rotated_background,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #

#"""Test 1
dataset = MNIST()


#"""
#

