"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as N
import warnings
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
from pylearn2.utils.rng import make_np_rng


class MNIST(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
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

    def __init__(self, which_set, center=False, shuffle=False,
                 one_hot=None, binarize=False, start=None,
                 stop=None, axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):
        self.args = locals()

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "There is no such thing as the MNIST validation set. MNIST"
                    "consists of 60,000 train examples and 10,000 test"
                    "examples. If you wish to use a validation set you should"
                    "divide the train set yourself. The pylearn2 dataset"
                    "implements and will only ever implement the standard"
                    "train / test split used in the literature.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/mnist/"
            if which_set == 'train':
                im_path = path + 'train-images-idx3-ubyte'
                label_path = path + 'train-labels-idx1-ubyte'
            else:
                assert which_set == 'test'
                im_path = path + 't10k-images-idx3-ubyte'
                label_path = path + 't10k-labels-idx1-ubyte'
            # Path substitution done here in order to make the lower-level
            # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
            # the Deep Learning Tutorials, or in another package).
            im_path = serial.preprocess(im_path)
            label_path = serial.preprocess(label_path)

            # Locally cache the files before reading them
            datasetCache = cache.datasetCache
            im_path = datasetCache.cache_file(im_path)
            label_path = datasetCache.cache_file(label_path)

            topo_view = read_mnist_images(im_path, dtype='float32')
            y = np.atleast_2d(read_mnist_labels(label_path)).T
        else:
            if which_set == 'train':
                size = 60000
            elif which_set == 'test':
                size = 10000
            else:
                raise ValueError(
                    'Unrecognized which_set value "%s".' % (which_set,) +
                    '". Valid values are ["train","test"].')
            topo_view = np.random.rand(size, 28, 28)
            y = np.random.randint(0, 10, (size, 1))

        if binarize:
            topo_view = (topo_view > 0.5).astype('float32')

        max_labels = 10
        if one_hot is not None:
            warnings.warn("the `one_hot` parameter is deprecated. To get "
                          "one-hot encoded targets, request that they "
                          "live in `VectorSpace` through the `data_specs` "
                          "parameter of MNIST's iterator method. "
                          "`one_hot` will be removed on or after "
                          "September 20, 2014.", stacklevel=2)

        m, r, c = topo_view.shape
        assert r == 28
        assert c == 28
        topo_view = topo_view.reshape(m, r, c, 1)

        if which_set == 'train':
            assert m == 60000
        elif which_set == 'test':
            assert m == 10000
        else:
            assert False

        if center:
            topo_view -= topo_view.mean(axis=0)

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for i in xrange(topo_view.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp
                # Note: slicing with i:i+1 works for one_hot=True/False
                tmp = y[i:i + 1].copy()
                y[i] = y[j]
                y[j] = tmp

        super(MNIST, self).__init__(topo_view=dimshuffle(topo_view), y=y,
                                    axes=axes, y_labels=max_labels)

        assert not N.any(N.isnan(self.X))

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
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

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


class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    one_hot : WRITEME
    """

    def __init__(self, which_set, center=False, one_hot=False):
        path = "${PYLEARN2_DATA_PATH}/mnist/mnist_rotation_back_image/" \
            + which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        self.one_hot = one_hot
        if one_hot:
            one_hot = N.zeros((y.shape[0], 10), dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i, y[i]] = 1.
            y = one_hot

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28, 28, 1))

        super(MNIST_rotated_background, self).__init__(
            X=X, y=y, view_converter=view_converter)

        assert not N.any(N.isnan(self.X))
