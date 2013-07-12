from pylearn.datasets import utlc
from pylearn2.datasets.exc import EnvironmentVariableError, NotInstalledError
import numpy as N
import os

class Avicenna(object):
    def __init__(self, which_set, standardize):
        if 'PYLEARN2_DATA_PATH' not in os.environ:
            raise NoDataPathError()
        if not os.path.exists(os.path.join(os.environ['PYLEARN2_DATA_PATH'], 'avicenna')):
            raise NotInstalledError() #XXX: check path

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
            union = N.concatenate([train,valid,test],axis=0)
            self.X -= union.mean(axis=0)
            std = union.std(axis=0)
            std[std < 1e-3] = 1e-3
            self.X /= std



    def get_design_matrix(self):
        return self.X
