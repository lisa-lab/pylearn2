import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import icml07

class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        orig = icml07.MNIST_rotated_background(n_train=10000,n_valid=2000,n_test=10000)

        sets = {
                'train' : orig.train,
                'valid' : orig.valid,
                'test'  : orig.test
            }

        X = N.cast['float32'](sets[which_set].x)
        y = sets[which_set].y


        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST_rotated_background,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #

#
