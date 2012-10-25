import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class CIFAR100(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False,
            gcn = None):

        assert which_set in ['train','test']

        path = "${PYLEARN2_DATA_PATH}/cifar100/cifar-100-python/"+which_set

        obj = serial.load(path)
        X = obj['data']

        assert X.max() == 255.
        assert X.min() == 0.

        X = N.cast['float32'](X)
        y = None #not implemented yet

        self.center = center

        if center:
            X -= 127.5

        self.gcn = gcn
        if gcn is not None:
            assert isinstance(gcn,float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
            X *= gcn

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        super(CIFAR100,self).__init__(X = X, y =y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

        self.y_fine = N.asarray(obj['fine_labels'])
        self.y_coarse = N.asarray(obj['coarse_labels'])

        self.y = self.y_fine


    def adjust_for_viewer(self, X):
        #assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(rval[i,:]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example = False):
        # if the scale is set based on the data, display X oring the scale determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i,:] /= np.abs(orig[i,:]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval
