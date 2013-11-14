"""TODO: module-level docstring."""
__authors__ = "Ian Goodfellow and Mehdi Mirza"
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
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.utils import safe_zip
from theano import config


def ensure_tables():
    """
    Makes sure tables module has been imported
    """

    global tables
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
                 rng=_default_seed, preprocessor = None, fit_preprocessor=False):
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
            An object for converting between the design matrix
            stored internally and the data that will be returned
            by iterators.
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
        """
        self.X = X
        self.y = y

        if topo_view is not None:
            assert view_converter is None
            self.set_topological_view(topo_view, axes)
        else:
            assert X is not None, ("DenseDesignMatrix needs to be provided "
                    "with either topo_view, or X")
            if view_converter is not None:
                self.view_converter = view_converter

                # Get the topo_space (usually Conv2DSpace) from the
                # view_converter
                if not hasattr(view_converter, 'topo_space'):
                    raise NotImplementedError("Not able to get a topo_space "
                            "from this converter: %s"
                            % view_converter)

                # self.X_topo_space stores a "default" topological space that
                # will be used only when self.iterator is called without a
                # data_specs, and with "topo=True", which is deprecated.
                self.X_topo_space = view_converter.topo_space
            else:
                self.X_topo_space = None

            # Update data specs, if not done in set_topological_view
            X_space = VectorSpace(dim=self.X.shape[1])
            X_source = 'features'
            if y is None:
                space = X_space
                source = X_source
            else:
                if self.y.ndim == 1:
                    dim = 1
                else:
                    dim = self.y.shape[-1]
                y_space = VectorSpace(dim=dim)
                y_source = 'targets'

                space = CompositeSpace((X_space, y_space))
                source = (X_source, y_source)
            self.data_specs = (space, source)
            self.X_space = X_space

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
        self._iter_data_specs = (self.X_space, 'features')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError("In DenseDesignMatrix.iterator, both "
                        "the `data_specs` argument and deprecated arguments "
                        "`topo` or `targets` were provided.",
                        (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are being "
                    "deprecated, and will be removed around November 7th, "
                    "2013. `data_specs` should be used instead.",
                    stacklevel=2)
            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                # self.iterator is called without a data_specs, and with
                # "topo=True", so we use the default topological space
                # stored in self.X_topo_space
                assert self.X_topo_space is not None
                X_space = self.X_topo_space
            else:
                X_space = self.X_space

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                assert self.y is not None
                y_space = self.data_specs[0].components[1]
                space = CompositeSpace((X_space, y_space))
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            data_specs = (space, source)
            convert = None

        else:
            if data_specs is None:
                data_specs = self._iter_data_specs

            # If there is a view_converter, we have to use it to convert
            # the stored data for "features" into one that the iterator
            # can return.
            space, source = data_specs
            if isinstance(space, CompositeSpace):
                sub_spaces = space.components
                sub_sources = source
            else:
                sub_spaces = (space,)
                sub_sources = (source,)

            convert = []
            for sp, src in safe_zip(sub_spaces, sub_sources):
                if (src == 'features' and
                        getattr(self, 'view_converter', None) is not None):
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(
                                   batch,
                                   space))
                else:
                    conv_fn = None
                convert.append(conv_fn)

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
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def get_data(self):
        """
        Returns all the data, as it is internally stored.

        The definition and format of these data are described in
        `self.get_data_specs()`.
        """
        if self.y is None:
            return self.X
        else:
            return (self.X, self.y)

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

        # To be able to unpickle older data after the addition of
        # the data_specs mechanism
        if not all(m in d for m in ('data_specs', 'X_space',
                                    '_iter_data_specs', 'X_topo_space')):
            X_space = VectorSpace(dim=self.X.shape[1])
            X_source = 'features'
            if self.y is None:
                space = X_space
                source = X_source
            else:
                y_space = VectorSpace(dim=self.y.shape[-1])
                y_source = 'targets'

                space = CompositeSpace((X_space, y_space))
                source = (X_source, y_source)

            self.data_specs = (space, source)
            self.X_space = X_space
            self._iter_data_specs = (X_space, X_source)

            view_converter = d.get('view_converter', None)
            if view_converter is not None:
                # Get the topo_space from the view_converter
                if not hasattr(view_converter, 'topo_space'):
                    raise NotImplementedError("Not able to get a topo_space "
                            "from this converter: %s"
                            % view_converter)

                # self.X_topo_space stores a "default" topological space that
                # will be used only when self.iterator is called without a
                # data_specs, and with "topo=True", which is deprecated.
                self.X_topo_space = view_converter.topo_space

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

    def get_formatted_view(self, mat, dspace):
        """
        Convert an array (or the entire dataset) to a destination space.

        Parameters
        ----------
        mat : ndarray, 2-dimensional
            An array containing a design matrix representation of training
            examples.

        dspace : Space
            A Space we want the data in mat to be formatted in. It can be
            a VectorSpace for a design matrix output, a Conv2DSpace for a
            topological output for instance. Valid values depend on the
            type of `self.view_converter`.
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_formatted_view on a dataset "
                            "that has no view converter")

        self.X_space.np_validate(mat)
        return self.view_converter.get_formatted_batch(mat, dspace)

    def get_weights_view(self, mat):
        """
        Return a view of mat in the topology preserving format.  Currently
        the same as get_topological_view.
        """

        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training
            examples.
        TODO: why is this parameter named 'V'?
        """
        assert not N.any(N.isnan(V))
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not N.any(N.isnan(self.X))

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            y_space = VectorSpace(dim=self.y.shape[-1])
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

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

    def restrict(self, start, stop):
        """
        Restricts the dataset to include only the examples
        in range(start, stop). Ignored if both arguments are None.
        """
        assert (start is None) == (stop is None)
        if start is None:
            return
        assert start >= 0
        assert stop > start
        assert stop <= self.X.shape[0]
        assert self.X.shape[0] == self.y.shape[0]
        self.X = self.X[start:stop, :]
        if self.y is not None:
            self.y = self.y[start:stop, :]
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == stop - start

    def convert_to_one_hot(self, min_class=0):
        """
        If y exists and is a vector of ints, converts it to a binary matrix
        Otherwise will raise some exception
        """

        if self.y is None:
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix with no labels.")

        if self.y.ndim != 1:
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't scalar.")

        if 'int' not in str(self.y.dtype):
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't integer-valued.")

        self.y = self.y - min_class

        if self.y.min() < 0:
            raise ValueError("We do not support negative classes. You can use"
                    "the min_class argument to remap negative classes to "
                    "positive values, but we require this to be done"
                    "explicitly so you are aware of the remapping.")
        # Note: we don't check that the minimum occurring class is exactly 0,
        # since this dataset could be just a small subset of a larger dataset
        # and may not contain all the classes.

        num_classes = self.y.max() + 1

        y = np.zeros((self.y.shape[0], num_classes))

        for i in xrange(self.y.shape[0]):
            y[i, self.y[i]] = 1

        self.y = y

        # Update self.data_specs with the updated dimension of self.y
        init_space, source = self.data_specs
        X_space, init_y_space = init_space.components
        new_y_space = VectorSpace(dim=num_classes)
        new_space = CompositeSpace((X_space, new_y_space))
        self.data_specs = (new_space, source)

    def adjust_for_viewer(self, X):
        return X / np.abs(X).max()

    def adjust_to_be_viewed_with(self, X, ref, per_example=None):
        if per_example is not None:
            warnings.warn("ignoring per_example")
        return np.clip(X / np.abs(ref).max(), -1., 1.)

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def set_view_converter_axes(self, axes):
        """
        Change the axes of the view_converter, if any.

        This function is only useful if you intend to call self.iterator
        without data_specs, and with "topo=True", which is deprecated.
        """
        assert self.view_converter is not None

        self.view_converter.set_axes(axes)
        # Update self.X_topo_space, which stores the "default"
        # topological space, which is the topological output space
        # of the view_converter
        self.X_topo_space = self.view_converter.topo_space


class DenseDesignMatrixPyTables(DenseDesignMatrix):
    """
    DenseDesignMatrix based on PyTables
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

        super(DenseDesignMatrixPyTables, self).__init__(X = X,
                                            topo_view = topo_view,
                                            y = y,
                                            view_converter = view_converter,
                                            axes = axes,
                                            rng = rng)
        ensure_tables()
        if not hasattr(self, 'filters'):
            self.filters = tables.Filters(complib='blosc', complevel=5)

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
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        warnings.warn("Overloading this method is not necessary with the new "
                     "interface change and this will be removed around November "
                     "7th 2013", stacklevel=2)

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError("In DenseDesignMatrix.iterator, both "
                        "the `data_specs` argument and deprecated arguments "
                        "`topo` or `targets` were provided.",
                        (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are being "
                    "deprecated, and will be removed around November 7th, "
                    "2013. `data_specs` should be used instead.",
                    stacklevel=2)
            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                # self.iterator is called without a data_specs, and with
                # "topo=True", so we use the default topological space
                # stored in self.X_topo_space
                assert self.X_topo_space is not None
                X_space = self.X_topo_space
            else:
                X_space = self.X_space

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                assert self.y is not None
                y_space = self.data_specs[0][1]
                space = (X_space, y_space)
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            data_specs = (space, source)
            _deprecated_interface = True
        else:
            _deprecated_interface = False

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
        if rng is None and mode.stochastic:
            rng = self.rng
        if data_specs is None:
            data_specs = self._iter_data_specs
        if _deprecated_interface:
            return FiniteDatasetIteratorPyTables(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple)
        else:
            return FiniteDatasetIterator(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple)

    def init_hdf5(self, path, shapes):
        """
        Initialize hdf5 file to be used ba dataset
        """

        x_shape, y_shape = shapes
        # make pytables
        ensure_tables()
        h5file = tables.openFile(path, mode = "w", title = "SVHN Dataset")
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
        h5file.createCArray(gcolumns, 'X', atom = atom, shape = x_shape,
                                title = "Data values", filters = self.filters)
        h5file.createCArray(gcolumns, 'y', atom = atom, shape = y_shape,
                                title = "Data targets", filters = self.filters)
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

    def resize(self, h5file, start, stop):
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

        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
        x = h5file.createCArray(gcolumns, 'X', atom = atom, shape = ((stop - start, data.X.shape[1])),
                            title = "Data values", filters = self.filters)
        y = h5file.createCArray(gcolumns, 'y', atom = atom, shape = ((stop - start, 10)),
                            title = "Data targets", filters = self.filters)
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
        self._update_topo_space()

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
        rval =  self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
            for axis in ('b', 0, 1, 'c')))

        return rval

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

    def get_formatted_batch(self, batch, dspace):
        """
        Reformat batch from the internal storage format into dspace.
        """
        if isinstance(dspace, VectorSpace):
            # If a VectorSpace is requested, batch should already be
            # in that space.
            dspace.np_validate(batch)
            return batch
        elif isinstance(dspace, Conv2DSpace):
            # design_mat_to_topo_view will return a batch formatted
            # in a Conv2DSpace, but not necessarily the right one.
            topo_batch = self.design_mat_to_topo_view(batch)
            if self.topo_space.axes != self.axes:
                warnings.warn("It looks like %s.axes has been changed "
                              "directly, please use the set_axes() method "
                              "instead." % self.__class__.__name__)
                self._update_topo_space()
            return self.topo_space.np_format_as(topo_batch, dspace)
        else:
            raise ValueError("%s does not know how to format a batch into "
                             "%s of type %s."
                             % (self.__class__.__name__, dspace, type(dspace)))

    def __setstate__(self, d):
        # Patch old pickle files that don't have the axes attribute.
        if 'axes' not in d:
            d['axes'] = ['b', 0, 1, 'c']
        self.__dict__.update(d)

        # Same for topo_space
        if 'topo_space' not in self.__dict__:
            self._update_topo_space()

    def _update_topo_space(self):
        """Update self.topo_space from self.shape and self.axes"""
        rows, cols, channels = self.shape
        self.topo_space = Conv2DSpace(shape=(rows, cols),
                                      num_channels=channels,
                                      axes=self.axes)

    def set_axes(self, axes):
        self.axes = axes
        self._update_topo_space()


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

def convert_to_one_hot(dataset, min_class=0):
    """
    Convenient way of accessing convert_to_one_hot from a yaml file
    """
    dataset.convert_to_one_hot(min_class=min_class)
    return dataset

def set_axes(dataset, axes):
    dataset.set_view_converter_axes(axes)
    return dataset
