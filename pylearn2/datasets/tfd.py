import numpy as np
import pickle
from pylearn2.datasets import dense_design_matrix

class TFD(dense_design_matrix.DenseDesignMatrix):

    def __init__(self):
        pass

    def set_fold(self, fold_index):
        assert fold_index in [0,1,2,3,4]


class UnsupervisedTFD(dense_design_matrix.DenseDesignMatrix):

    root = '/data/lisatmp/rifaisal/TFD/unsupervised/'

    def __init__(self, center = True):

        X = np.zeros((0, 2304))
         
        for idx in xrange(12):

            fname = self.root + 'TFD_unsupervised_train_unlabeled%i.pkl' % idx
            print 'loading subset %s...' % fname

            fd = open(fname)
            data = pickle.load(fd)
            fd.close()

            X = np.vstack((X, data[0]))
            del data

        if center:
            X -= 127.5
        view_converter = dense_design_matrix.DefaultViewConverter((48,48,1))
        super(UnsupervisedTFD, self).__init__(X=X, view_converter=view_converter)
