import numpy as N
from framework.datasets import dense_design_matrix
import pylearn.datasets.MNIST as i_hate_python

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
            assert False

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #

#
