"""
.. todo::

    WRITEME
"""
import os, cPickle, logging
_logger = logging.getLogger(__name__)

import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
import warnings

from pylearn2.datasets.preprocessing import GlobalContrastNormalization, \
    TorontoPreprocessor, CenterPreprocessor, RescalePreprocessor
from pylearn2.datasets.preprocessing import Pipeline


class CIFAR10(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    rescale : WRITEME
    gcn : WRITEME
    one_hot : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """
    def __init__(self, which_set, center = False, rescale = False, gcn = None,
            one_hot = False, start = None, stop = None, axes=('b', 0, 1, 'c'),
            toronto_prepro = False, preprocessor = None, preprocessors=[]):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.center = center
        self.rescale = rescale
        self.toronto_prepro = toronto_prepro
        self.gcn = gcn
        self.preprocessors = preprocessors

        self.validate_options()

        self.axes = axes

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


        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3), axes)

        super(CIFAR10, self).__init__(X=X, y=y, view_converter=view_converter)

        # TODO: should we check that rescale is not enabled when doing toronto_prepro
        # Also, if we can do rescale with toronto_prepro, should we make sure
        # that rescale is done before toronto_prepro? Should we check that
        # center is done before rescale?
        for preproc in self.preprocessors:
            class_name = preproc.__class__.__name__
            can_fit = class_name == TorontoPreprocessor.__name__
            if can_fit:
                if which_set == 'train':
                    preproc.apply(self, can_fit=can_fit)
                    continue
                else:
                    preproc.apply(CIFAR10(which_set='train'), can_fit=can_fit)
            preproc.apply(self)

        if start is not None:
            # This needs to come after the prepro so that it doesn't change the pixel
            # means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= self.X.shape[0]
            self.X = self.X[start:stop, :]
            self.y = self.y[start:stop]
            assert self.X.shape[0] == self.y.shape[0]

        if which_set == 'test':
            assert self.X.shape[0] == 10000

        assert not np.any(np.isnan(self.X))

        # TODO: should we check that preprocessor is not already in `preprocessors`?
        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
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

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return CIFAR10(which_set='test', center=self.center, rescale=self.rescale, gcn=self.gcn,
                one_hot=self.one_hot, toronto_prepro=self.toronto_prepro, axes=self.axes)


    @classmethod
    def _unpickle(cls, file):
        """
        .. todo::

            What is this? why not just use serial.load like the CIFAR-100
            class? Whoever wrote it shows up as "unknown" in git blame.
        """
        from pylearn2.utils import string_utils
        fname = os.path.join(
                string_utils.preprocess('${PYLEARN2_DATA_PATH}'),
                'cifar10',
                'cifar-10-batches-py',
                file)
        if not os.path.exists(fname):
            raise IOError(fname+" was not found. You probably need to download "
                    "the CIFAR-10 dataset by using the download script in pylearn2/scripts/datasets/download_cifar10.sh "
                    "or manually from http://www.cs.utoronto.ca/~kriz/cifar.html")
        _logger.info('loading file %s' % fname)
        fo = open(fname, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict


    def validate_options(self):
        """
        Performs the following validations on the constructor's arguments:
            1) check that `preprocessors` does not have any duplicated
               preprocessors. If there is duplication, a warning is issued.
               TODO: should we delete the duplicated preprocessor from `preprocessors`?
            2) if an option (center, rescale, gcn or toronto_prepro) was
               specified, then the corresponding preprocessor from the
               Preprocessor class is instantiated and added to `preprocessors`
        """

        ### Check that the list of preprocessors `processors` does not have
        ### duplicated preprocessors
        self.set_preproc = set()
        for preproc in self.preprocessors:
            class_name = preproc.__class__.__name__
            if class_name in self.set_preproc:
                warnings.warn("The preprocessor %s is found more than once in the "
                "list of preprocessors."%class_name)
            else:
                self.set_preproc.add(class_name)

        ### Check that any of the options are specified. Issue a warning
        ### and instantiate the corresponding preprocessor if that's the case
        if self.center:
            warnings.warn("The option center is deprecated. The Center "
                        "preprocessor should be used instead.")
            if CenterPreprocessor.__name__ in self.set_preproc:
                warnings.warn("The option center was specified but the list "
                            "preprocessors already contains CenterPreprocessor.")
            else:
                self.preprocessors.append(RescalePreprocessor())

        if self.rescale:
            warnings.warn("The option rescale is deprecated. The Rescale "
                        "preprocessor should be used instead.")
            if RescalePreprocessor.__name__ in self.set_preproc:
                warnings.warn("The option rescale was specified but the list "
                            "preprocessors already contains RescalePreprocessor.")
            else:
                self.preprocessors.append(RescalePreprocessor())

        if self.gcn:
            warnings.warn("The option gcn is deprecated. The "
                        "GlobalContrastNormalization preprocessor should be "
                        "used instead.")
            if GlobalContrastNormalization.__name__ in self.set_preproc:
                warnings.warn("The option gcn was specified but the list "
                            "preprocessors already contains "
                            "GlobalContrastNormalization.")
            else:
                self.preprocessors.append(GlobalContrastNormalization(scale=float(self.gcn)))

        if self.toronto_prepro:
            warnings.warn("The option toronto_preproc is deprecated. The "
                        "TorontoPreprocessor preprocessor should be used "
                        "instead.")
            assert not self.center
            assert not self.gcn
            if TorontoPreprocessor.__name__ in self.set_preproc:
                warnings.warn("The option toronto_prepro was specified but the "
                            "list preprocessors already contains a "
                            "TorontoPreprocessor.")
            else:
                self.preprocessors.append(TorontoPreprocessor())
