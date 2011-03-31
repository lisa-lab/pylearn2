import numpy as N
from framework.datasets import dense_design_matrix
from pylearn.datasets import cifar10

class CIFAR10(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set):
        orig = cifar10.cifar10()

        Xs = {
                'train' : orig.train.x,
                'valid' : orig.valid.x,
                'test'  : orig.test.x
            }

        X = Xs[which_set]

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        super(CIFAR10,self).__init__(X = X, view_converter = view_converter)
    #

#
