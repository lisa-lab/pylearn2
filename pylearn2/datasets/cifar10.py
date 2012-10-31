import os, cPickle, logging
_logger = logging.getLogger('pylearn2.datasets.cifar10')

import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix

class CIFAR10(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False, rescale = False, gcn = None,
            one_hot = False):

        # note: there is no such thing as the cifar10 validation set;
        # quit pretending that there is.

        # we define here:
        dtype  = 'uint8'
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest  = 10000

        # we also expose the following details:
        self.img_shape = (3,32,32)
        self.img_size = N.prod(self.img_shape)
        self.n_classes = 10
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog','horse','ship','truck']

        # prepare loading
        fnames = ['data_batch_%i' % i for i in range(1,6)]
        lenx = N.ceil((ntrain + nvalid) / 10000.)*10000
        x = N.zeros((lenx,self.img_size), dtype=dtype)
        y = N.zeros(lenx, dtype=dtype)

        # load train data
        nloaded = 0
        for i, fname in enumerate(fnames):
            data = CIFAR10._unpickle(fname)
            x[i*10000:(i+1)*10000, :] = data['data']
            y[i*10000:(i+1)*10000] = data['labels']
            nloaded += 10000
            if nloaded >= ntrain + nvalid + ntest: break;

        # load test data
        data = CIFAR10._unpickle('test_batch')

        # process this data
        Xs = {
                'train' : x[0:ntrain],
                'test'  : data['data'][0:ntest]
            }

        Ys = {
                'train' : y[0:ntrain],
                'test'  : data['labels'][0:ntest]
            }

        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if isinstance(y,list):
            y = np.asarray(y)

        if which_set == 'test':
            assert y.shape[0] == 10000

        self.one_hot = one_hot
        if one_hot:
            one_hot = np.zeros((y.shape[0],10),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale

        self.gcn = gcn
        if gcn is not None:
            assert isinstance(gcn,float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
            X *= gcn

        if which_set == 'test':
            assert X.shape[0] == 10000

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        super(CIFAR10,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

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


    @classmethod
    def _unpickle(cls, file):
        """
        TODO: wtf is this? why not just use serial.load like the CIFAR-100 class?
        whoever wrote it shows up as "unknown" in git blame
        """
        from pylearn2.utils import string_utils
        fname = os.path.join(
                string_utils.preprocess('${PYLEARN2_DATA_PATH}'),
                'cifar10',
                'cifar-10-batches-py',
                file)
        if not os.path.exists(fname):
            raise IOError(fname+" was not found. You probably need to download "
                    " the CIFAR-10 dataset from http://www.cs.utoronto.ca/~kriz/cifar.html")
        _logger.info('loading file %s' % fname)
        fo = open(fname, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
