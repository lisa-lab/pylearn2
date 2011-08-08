import numpy as N
from scipy import io
from pylearn2.datasets import dense_design_matrix
import theano
floatX = theano.config.floatX


class MatlabDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path, which_set):
        Xs = io.loadmat(path)
        X = Xs[which_set]
        super(MatlabDataset, self).__init__(X=N.cast[floatX](X))
        assert not N.any(N.isnan(self.X))
