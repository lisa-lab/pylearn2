import numpy as N
from scipy import io
from framework.datasets import dense_design_matrix

class MatlabDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path, which_set):

        Xs = io.loadmat(path)

        X = Xs[which_set]

        super(MatlabDataset,self).__init__(X = N.cast['float32'](X) )

        assert not N.any(N.isnan(self.X))
    #

#
