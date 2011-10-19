#The dataset for the NIPS 2011 Transfer Learning Challenge
import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.string import preprocess

class TL_Challenge(dense_design_matrix.DenseDesignMatrix):
     def __init__(self, which_set, center = False):

        assert which_set in ['train','unlabeled']

        path = "${PYLEARN2_DATA_PATH}/TLChallenge"

        if which_set == 'train':
            path += '/training/training-data.dat'
        else:
            path += '/unlabelled_tiny.dat'

        path = preprocess(path)

        X = N.fromfile(path, dtype=N.uint8, sep=' ')

        X = X.reshape(X.shape[0]/(32*32*3), 32*32* 3, order='F')

        assert X.max() == 255
        assert X.min() == 0

        X = N.cast['float32'](X)
        y = None #not implemented yet

        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        X = view_converter.design_mat_to_topo_view(X)

        X = N.transpose(X,(0,2,1,3))

        X = view_converter.topo_view_to_design_mat(X)

        super(TL_Challenge,self).__init__(X = X, y =y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #

#
