"""
The ZCA Dataset class.

This is basically a prototype for a more general idea of being
able to invert preprocessors and view data in more than one
format. This should be expected to change, but had to go in
pylearn2 to support pylearn2/scripts/papers/maxout
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.datasets import control

class ZCA_Dataset(DenseDesignMatrix):
    """
    A Dataset that was created by ZCA whitening a DenseDesignMatrix.
    Supports viewing the data both in the new ZCA whitened space and
    mapping examples back to the original space.
    """

    def get_test_set(self):
        yaml = self.preprocessed_dataset.yaml_src
        yaml = yaml.replace('train', 'test')
        args = {}
        args.update(self.args)
        del args['self']
        args['start'] = None
        args['stop'] = None
        args['preprocessed_dataset'] = yaml_parse.load(yaml)
        return ZCA_Dataset(**args)

    def __init__(self,
            preprocessed_dataset,
            preprocessor,
            convert_to_one_hot = True,
            start = None,
            stop = None,
            axes = ['b', 0, 1, 'c']):

        self.args = locals()

        self.preprocessed_dataset = preprocessed_dataset
        self.preprocessor = preprocessor
        self.rng = self.preprocessed_dataset.rng

        self.y = preprocessed_dataset.y
        assert self.y is not None
        if convert_to_one_hot:
            if not ( self.y.min() == 0):
                raise AssertionError("Expected y.min == 0 but y.min == "+str(self.y.min()))
            nclass = self.y.max() + 1
            y = np.zeros((self.y.shape[0], nclass), dtype='float32')
            for i in xrange(self.y.shape[0]):
                y[i,self.y[i]] = 1.
            self.y = y
            assert self.y is not None

        if control.get_load_data():
            if start is not None:
                self.X = preprocessed_dataset.X[start:stop,:]
                self.y = self.y[start:stop,:]
                assert self.X.shape[0] == stop-start
            else:
                self.X = preprocessed_dataset.X
            assert self.y is not None
        else:
            self.X = None
        if self.X is not None:
            assert self.y.shape[0] == self.X.shape[0]
        self.view_converter = preprocessed_dataset.view_converter

        #self.mn = self.X.min()
        #self.mx = self.X.max()
        print 'inverting...'
        preprocessor.invert()
        print '...done inverting'

        self.view_converter.axes = axes

    def has_targets(self):
        return self.preprocessed_dataset.has_targets()

    def adjust_for_viewer(self, X):

        #rval = X - self.mn
        #rval /= (self.mx-self.mn)

        #rval *= 2.
        #rval -= 1.
        rval = X.copy()

        #rval = np.clip(rval,-1.,1.)


        for i in xrange(rval.shape[0]):
            rval[i,:] /= np.abs(rval[i,:]).max() + 1e-12

        return rval

    def adjust_to_be_viewed_with(self, X, other, per_example = False):

        #rval = X - self.mn
        #rval /= (self.mx-self.mn)

        #rval *= 2.
        #rval -= 1.


        rval = X.copy()

        if per_example:
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(other[i,:]).max()
        else:
            rval /= np.abs(other).max()

        rval = np.clip(rval,-1.,1.)

        return rval

    def mapback_for_viewer(self, X):

        assert X.ndim == 2
        rval = self.preprocessor.inverse(X)
        rval = self.preprocessed_dataset.adjust_for_viewer(rval)

        return rval

    def mapback(self, X):
        return self.preprocessor.inverse(X)

