import numpy as N
from pylearn2.datasets import dense_design_matrix

class Wiskott(dense_design_matrix.DenseDesignMatrix):
    def __init__(self):

        X = 1. - N.load("/data/lisa/data/wiskott/wiskott_fish_layer0_15_standard_64x64_shuffled.npy")


        view_converter = dense_design_matrix.DefaultViewConverter((64,64,1))

        super(Wiskott,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #
#
