"""
Binarized, unlabeled version of the MNIST dataset, as referenced in

    On the Quantitative Analysis of Deep Belief Networks
    Salakhutdinov and Murray
    http://www.mit.edu/~rsalakhu/papers/dbn_ais.pdf
    The MNIST database of handwritten digits
    LeCun and Cortes
    http://yann.lecun.com/exdb/mnist/
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


class BinarizedMNIST(DenseDesignMatrix):
    """
    Parameters
    ----------
    which_set : str
        Must be in ['train', 'valid', 'test']
    center : WRITEME
    shuffle : WRITEME
    one_hot : WRITEME
    binarize : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    fit_preprocessor : WRITEME
    fit_test_preprocessor : WRITEME
    """

    def __init__(self, which_set, center=False, shuffle=False, one_hot=None,
                 binarize=False, start=None, stop=None, axes=['b', 0, 1, 'c'],
                 preprocessor=None, fit_preprocessor=False,
                 fit_test_preprocessor=False):
        self.args = locals()

        if which_set not in ['train', 'valid', 'test']:
            raise ValueError('Unrecognized which_set value "%s".' %
                             (which_set,) + '". Valid values are ' +
                             '["train", "valid", "test"].')

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/binarized_mnist/binarized_mnist_" + \
                   which_set + ".npy"
            im_path = serial.preprocess(path)

            # Locally cache the files before reading them
            datasetCache = cache.datasetCache
            im_path = datasetCache.cache_file(im_path)

            X = serial.load(im_path)
        else:
            if which_set == 'train':
                size = 50000
            else:
                size = 10000
            X = numpy.random.rand(size, 28 ** 2)

        m, d = X.shape
        assert d == 28 ** 2
        if which_set == 'train':
            assert m == 50000
        else:
            assert m == 10000

        if center:
            X -= X.mean(axis=0)

        if shuffle:
            self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
            for i in xrange(X.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = X[i, :].copy()
                X[i, :] = X[j, :]
                X[j, :] = tmp

        super(BinarizedMNIST, self).__init__(X=X)

        assert not numpy.any(numpy.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        return N.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        .. todo::

            WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return MNIST(**args)
