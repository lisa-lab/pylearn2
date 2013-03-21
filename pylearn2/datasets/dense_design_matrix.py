"""TODO: module-level docstring."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import functools

import warnings
import numpy as np
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    FiniteDatasetIteratorPyTables,
    resolve_iterator_class
)
N = np
import copy
# Don't import tables initially, since it might not be available
# everywhere.
tables = None


from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import control
from theano import config

def ensure_tables():
    """
    Makes sure tables module has been imported
    """

    if tables is None:
        import tables


class DenseDesignMatrix(Dataset):
    """
    A class for representing datasets that can be stored as a dense design
    matrix, such as MNIST or CIFAR10.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, axes = ('b', 0, 1, 'c'),
                 rng=_default_seed):
        """
        Parameters
        ----------

        X : ndarray, 2-dimensional, optional
            Should be supplied if `topo_view` is not. A design
            matrix of shape (number examples, number features)
            that defines the dataset.
        topo_view : ndarray, optional
            Should be supplied if X is not.  An array whose first
            dimension is of length number examples. The remaining
            dimensions are xamples with topological significance,
            e.g. for images the remaining axes are rows, columns,
            and channels.
        y : ndarray, 1-dimensional(?), optional
            Labels or targets for each example. The semantics here
            are not quite nailed down for this yet.
        view_converter : object, optional
            An object for converting between design matrices and
            topological views. Currently DefaultViewConverter is
            the only type available but later we may want to add
            one that uses the retina encoding that the U of T group
            uses.
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
        """
        self.X = X
        if view_converter is not None:
            assert topo_view is None
            self.view_converter = view_converter
        else:
            if topo_view is not None:
                self.set_topological_view(topo_view, axes)
        self.y = y
        self.compress = False
        self.design_loc = None
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_targets = False

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None):

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)
        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if topo is None:
            topo = getattr(self, '_iter_topo', False)
        if targets is None:
            targets = getattr(self, '_iter_targets', False)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     topo, targets)

    def use_design_loc(self, path):
        """
        When pickling, save the design matrix to path as a .npy file rather
        than pickling the design matrix along with the rest of the dataset
        object. This avoids pickle's unfortunate behavior of using 2X the RAM
        when unpickling.

        TODO: Get rid of this logic, use custom array-aware picklers (joblib,
        custom pylearn2 serialization format).
        """
        self.design_loc = path

    def get_topo_batch_axis(self):
        return self.view_converter.axes.index('b')

    def enable_compression(self):
        """
        If called, when pickled the dataset will be saved using only
        8 bits per element.

        TODO: Not sure this should be implemented as something a base dataset
        does. Perhaps as a mixin that specific datasets (i.e. CIFAR10) inherit
        from.
        """
        self.compress = True

    def __getstate__(self):
        rval = copy.copy(self.__dict__)
        # TODO: Not sure this should be implemented as something a base dataset
        # does. Perhaps as a mixin that specific datasets (i.e. CIFAR10)
        # inherit from.
        if self.compress:
            rval['compress_min'] = rval['X'].min(axis=0)
            # important not to do -= on this line, as that will modify the
            # original object
            rval['X'] = rval['X'] - rval['compress_min']
            rval['compress_max'] = rval['X'].max(axis=0)
            rval['compress_max'][rval['compress_max'] == 0] = 1
            rval['X'] *= 255. / rval['compress_max']
            rval['X'] = N.cast['uint8'](rval['X'])

        if self.design_loc is not None:
            # TODO: Get rid of this logic, use custom array-aware picklers
            # (joblib, custom pylearn2 serialization format).
            N.save(self.design_loc, rval['X'])
            del rval['X']

        return rval

    def __setstate__(self, d):

        if d['design_loc'] is not None:
            if control.get_load_data():
                d['X'] = N.load(d['design_loc'])
            else:
                d['X'] = None

        if d['compress']:
            X = d['X']
            mx = d['compress_max']
            mn = d['compress_min']
            del d['compress_max']
            del d['compress_min']
            d['X'] = 0
            self.__dict__.update(d)
            if X is not None:
                self.X = N.cast['float32'](X) * mx / 255. + mn
            else:
                self.X = None
        else:
            self.__dict__.update(d)

    def _apply_holdout(self, _mode="sequential", train_size=0, train_prop=0):
        """
          This function splits the dataset according to the number of
          train_size if defined by the user with respect to the mode provided
          by the user. Otherwise it will use the
          train_prop to divide the dataset into a training and holdout
          validation set. This function returns the training and validation
          dataset.

          Parameters
          -----------
          train_size: The number of examples that will be assigned to
          the training dataset.
          train_prop: Proportion of training dataset split.
        """

        train = None
        valid = None
        if train_size !=0:
            dataset_iter = self.iterator(mode=_mode,
                    batch_size=(self.num_examples - train_size),
                    num_batches=2)
            train = dataset_iter.next()
            valid = dataset_iter.next()
        elif train_prop !=0:
            size = np.ceil(self.num_examples * train_prop)
            dataset_iter = self.iterator(mode=_mode,
                    batch_size=(self.num_examples - size))
            train = dataset_iter.next()
            valid = dataset_iter.next()
        else:
            raise ValueError("Initialize either split ratio and split size to non-zero value.")
        return (train, valid)

    def split_dataset_nfolds(self, nfolds=0):
        """
          This function splits the dataset into to the number of n folds
          given by the user. Returns an array of folds.

          Parameters
          -----------
          nfolds: The number of folds for the  the validation set.
        """

        folds_iter = self.iterator(mode="sequential", num_batches=nfolds)
        folds = list(folds_iter)
        return folds

    def split_dataset_holdout(self, train_size=0, train_prop=0):
        """
          This function splits the dataset according to the number of
          train_size if defined by the user. Otherwise it will use the
          train_prop to divide the dataset into a training and holdout
          validation set. This function returns the training and validation
          dataset.

          Parameters
          -----------
          train_size: The number of examples that will be assigned to
          the training dataset.
          train_prop: Proportion of dataset split.
        """
        return self._apply_holdout("sequential", train_size, train_prop)

    def bootstrap_nfolds(self, nfolds, rng=None):
        """
          This function splits the dataset using the random_slice and into the
          n folds. Returns the folds.

          Parameters
          -----------
          nfolds: The number of folds for the  dataset.
          rng: Random number generation class to be used.
        """

        folds_iter = self.iterator(mode="random_slice", num_batches=nfolds, rng=rng)
        folds = list(folds_iter)
        return folds

    def bootstrap_holdout(self, train_size=0, train_prop=0, rng=None):
        """
          This function splits the dataset according to the number of
          train_size defined by the user.

          Parameters
          -----------
          train_size: The number of examples that will be assigned to
          the training dataset.
          nfolds: The number of folds for the  the validation set.
          rng: Random number generation class to be used.
        """
        return self._apply_holdout("random_slice", train_size, train_prop)

    def get_stream_position(self):
        """
        If we view the dataset as providing a stream of random examples to
        read, the object returned uniquely identifies our current position in
        that stream.
        """
        return copy.copy(self.rng)

    def set_stream_position(self, pos):
        """
        Return to a state specified by an object returned from
        get_stream_position.
        """
        self.rng = copy.copy(pos)

    def restart_stream(self):
        """
        Return to the default initial state of the random example stream.
        """
        self.reset_RNG()

    def reset_RNG(self):
        """
        Restore the default seed of the rng used for choosing random
        examples.
        """

        if 'default_rng' not in dir(self):
            self.default_rng = N.random.RandomState([17, 2, 946])
        self.rng = copy.copy(self.default_rng)

    def apply_preprocessor(self, preprocessor, can_fit=False):
        preprocessor.apply(self, can_fit)

    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

            This parameter is not named X because X is generally used to
            refer to the design matrix for the current problem. In this
            case we want to make it clear that `mat` need not be the design
            matrix defining the dataset.
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset "
                            "that has no view converter")
        if mat is None:
            mat = self.X
        return self.view_converter.design_mat_to_topo_view(mat)

    def get_weights_view(self, mat):
        """
        Return a view of mat in the topology preserving format.  Currently
        the same as get_topological_view.
        """

        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat)

    def set_topological_view(self, V, axes = ('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
        TODO: why is this parameter named 'V'?
        """
        assert not N.any(N.isnan(V))
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        assert not N.any(N.isnan(self.X))

    def get_design_matrix(self, topo=None):
        """
        Return topo (a batch of examples in topology preserving format),
        in design matrix format.

        Parameters
        ----------
        topo : ndarray, optional
            An array containing a topological representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
        """
        if topo is not None:
            if self.view_converter is None:
                raise Exception("Tried to convert from topological_view to "
                                "design matrix using a dataset that has no "
                                "view converter")
            return self.view_converter.topo_view_to_design_mat(topo)

        return self.X

    def set_design_matrix(self, X):
        assert len(X.shape) == 2
        assert not N.any(N.isnan(X))
        self.X = X

    def get_targets(self):
        return self.y

    @property
    def num_examples(self):
        return self.X.shape[0]

    def get_batch_design(self, batch_size, include_labels=False):
        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                raise ValueError("Requested "+str(batch_size)+" examples"
                    "from a dataset containing only "+str(self.X.shape[0]))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        rx = np.cast[config.floatX](rx)
        return rx

    def get_batch_topo(self, batch_size, include_labels = False):

        if include_labels:
            batch_design, labels = self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels

        return rval

    def view_shape(self):
        return self.view_converter.view_shape()

    def weights_view_shape(self):
        return self.view_converter.weights_view_shape()

    def has_targets(self):
         return self.y is not None

class DenseDesignMatrixPyTables(DenseDesignMatrix):
    """
    DenseDesignMatrix based on PyTables
    """

    ensure_tables()
    filters = tables.Filters(complib='blosc', complevel=5)

    def set_design_matrix(self, X, start = 0):
        assert len(X.shape) == 2
        assert not N.any(N.isnan(X))
        DenseDesignMatrixPyTables.fill_hdf5(file = self.h5file,
                                            data_x = X,
                                            start = start)

    def set_topological_view(self, V, axes = ('b', 0, 1, 'c'), start = 0):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
        TODO: why is this parameter named 'V'?
        """
        assert not N.any(N.isnan(V))
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        X = self.view_converter.topo_view_to_design_mat(V)
        assert not N.any(N.isnan(X))
        DenseDesignMatrixPyTables.fill_hdf5(file = self.h5file,
                                            data_x = X,
                                            start = start)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None):

        # TODO: Refactor, deduplicate with DenseDesignMatrix.iterator
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)
        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if topo is None:
            topo = getattr(self, '_iter_topo', False)
        if targets is None:
            targets = getattr(self, '_iter_targets', False)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIteratorPyTables(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     topo, targets)

    @staticmethod
    def init_hdf5(path, shapes):
        """
        Initialize hdf5 file to be used ba dataset
        """

        x_shape, y_shape = shapes
        # make pytables
        ensure_tables()
        h5file = tables.openFile(path, mode = "w", title = "SVHN Dataset")
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = tables.Float64Atom() if config.floatX == 'flaot32' else tables.Float32Atom()
        filters = DenseDesignMatrixPyTables.filters
        h5file.createCArray(gcolumns, 'X', atom = atom, shape = x_shape,
                                title = "Data values", filters = filters)
        h5file.createCArray(gcolumns, 'y', atom = atom, shape = y_shape,
                                title = "Data targets", filters = filters)
        return h5file, gcolumns

    @staticmethod
    def fill_hdf5(file, data_x, data_y = None, node = None, start = 0, batch_size = 5000):
        """
        PyTables tends to crash if you write large data on them at once.
        This function write data on file in batches

        start: the start index to write data
        """

        if node is None:
            node = file.getNode('/', 'Data')

        data_size = data_x.shape[0]
        last = np.floor(data_size / float(batch_size)) * batch_size
        for i in xrange(0, data_size, batch_size):
            stop = i + np.mod(data_size, batch_size) if i >= last else i + batch_size
            assert len(range(start + i, start + stop)) == len(range(i, stop))
            assert (start + stop) <= (node.X.shape[0])
            node.X[start + i: start + stop, :] = data_x[i:stop, :]
            if data_y is not None:
                 node.y[start + i: start + stop, :] = data_y[i:stop, :]

            file.flush()

    @staticmethod
    def resize(h5file, start, stop):
        ensure_tables()
        # TODO is there any smarter and more efficient way to this?

        data = h5file.getNode('/', "Data")
        try:
            gcolumns = h5file.createGroup('/', "Data_", "Data")
        except tables.exceptions.NodeError:
            h5file.removeNode('/', "Data_", 1)
            gcolumns = h5file.createGroup('/', "Data_", "Data")

        start = 0 if start is None else start
        stop = gcolumns.X.nrows if stop is None else stop

        atom = tables.Float64Atom() if config.floatX == 'flaot32' else tables.Float32Atom()
        filters = DenseDesignMatrixPyTables.filters
        x = h5file.createCArray(gcolumns, 'X', atom = atom, shape = ((stop - start, data.X.shape[1])),
                            title = "Data values", filters = filters)
        y = h5file.createCArray(gcolumns, 'y', atom = atom, shape = ((stop - start, 10)),
                            title = "Data targets", filters = filters)
        x[:] = data.X[start:stop]
        y[:] = data.y[start:stop]

        h5file.removeNode('/', "Data", 1)
        h5file.renameNode('/', "Data", "Data_")
        h5file.flush()
        return h5file, gcolumns

class DefaultViewConverter(object):
    def __init__(self, shape, axes = ('b', 0, 1, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes

    def view_shape(self):
        return self.shape

    def weights_view_shape(self):
        return self.shape

    def design_mat_to_topo_view(self, X):
        assert len(X.shape) == 2
        batch_size = X.shape[0]

        channel_shape = [batch_size, self.shape[0], self.shape[1], 1]
        dimshuffle_args = [('b', 0, 1, 'c').index(axis) for axis in self.axes]
        if self.shape[-1] * self.pixels_per_channel != X.shape[1]:
            raise ValueError('View converter with ' + str(self.shape[-1]) +
                             ' channels and ' + str(self.pixels_per_channel) +
                             ' pixels per channel asked to convert design'
                             ' matrix with ' + str(X.shape[1]) + ' columns.')
        start = lambda i: self.pixels_per_channel * i
        stop = lambda i: self.pixels_per_channel * (i + 1)
        channels = [X[:, start(i):stop(i)].reshape(*channel_shape).transpose(*dimshuffle_args)
                    for i in xrange(self.shape[-1])]

        channel_idx = self.axes.index('c')
        rval = np.concatenate(channels, axis=channel_idx)
        assert len(rval.shape) == len(self.shape) + 1
        return rval

    def design_mat_to_weights_view(self, X):
        return self.design_mat_to_topo_view(X)

    def topo_view_to_design_mat(self, V):

        V = V.transpose(self.axes.index('b'), self.axes.index(0),
                self.axes.index(1), self.axes.index('c'))

        num_channels = self.shape[-1]
        if N.any(N.asarray(self.shape) != N.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by ' + str(self.shape) +
                             ' given tensor of shape ' + str(V.shape))
        batch_size = V.shape[0]

        rval = N.zeros((batch_size, self.pixels_per_channel * num_channels),
                       dtype=V.dtype)

        for i in xrange(num_channels):
            ppc = self.pixels_per_channel
            rval[:, i * ppc:(i + 1) * ppc] = V[..., i].reshape(batch_size, ppc)
        assert rval.dtype == V.dtype

        return rval

    def __setstate__(self, d):
        # Patch old pickle files that don't have the axes attribute.
        if 'axes' not in d:
            d['axes'] = ['b', 0, 1, 'c']
        self.__dict__.update(d)

def from_dataset(dataset, num_examples):
    try:

        V, y = dataset.get_batch_topo(num_examples, True)

    except:

        if isinstance(dataset, DenseDesignMatrix) and dataset.X is None and not control.get_load_data():
                warnings.warn("from_dataset wasn't able to make subset of dataset, using the whole thing")
                return DenseDesignMatrix(X = None, view_converter = dataset.view_converter)
                #This patches a case where control.get_load_data() is false so dataset.X is None
                #This logic should be removed whenever we implement lazy loading
        raise

    rval =  DenseDesignMatrix(topo_view=V, y=y)
    rval.adjust_for_viewer = dataset.adjust_for_viewer

    return rval

def dataset_range(dataset, start, stop):

    if dataset.X is None:
        return DenseDesignMatrix(X = None, y = None, view_converter = dataset.view_converter)
    X = dataset.X[start:stop, :].copy()
    if dataset.y is None:
        y = None
    else:
        if dataset.y.ndim == 2:
            y = dataset.y[start:stop,:].copy()
        else:
            y = dataset.y[start:stop].copy()
        assert X.shape[0] == y.shape[0]
    assert X.shape[0] == stop - start
    topo = dataset.get_topological_view(X)
    rval = DenseDesignMatrix(topo_view = topo, y = y)
    rval.adjust_for_viewer = dataset.adjust_for_viewer
    return rval

