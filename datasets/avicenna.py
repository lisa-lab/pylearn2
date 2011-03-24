from pylearn.datasets import utlc
import numpy as N

class Avicenna:
    def __init__(self, which_set, standardize):

        #train, valid, test = N.random.randn(50,50), N.random.randn(50,50), N.random.randn(50,50)
        #print "avicenna hacked to load small random data instead of actual data"
        train, valid, test = utlc.load_ndarray_dataset('avicenna')

        if which_set == 'train':
            self.X = train
        elif which_set == 'valid':
            self.X = valid
        elif which_set == 'test':
            self.X = test
        else:
            assert False

        if standardize:
            self.X -= train.mean(axis=0)
            std = train.std(axis=0)
            std[std < 1e-3] = 1e-3
            self.X /= std

    def get_design_matrix(self):
        return self.X
