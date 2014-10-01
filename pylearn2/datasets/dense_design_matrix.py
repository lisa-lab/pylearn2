"""
The DenseDesignMatrix class and related code. Functionality for representing
data that can be described as a dense matrix (rather than a sparse matrix)
with each row containing an example and each column corresponding to a
different feature. DenseDesignMatrix also supports other "views" of the data,
for example a dataset of images can be viewed either as a matrix of flattened
images or as a stack of 2D multi-channel images. However, the images must all
be the same size, so that each image may be mapped to a matrix row by the same
transformation.
"""
__authors__ = "Ian Goodfellow and Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import functools

import logging
import warnings

import numpy as np

from pylearn2.datasets import cache
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

import copy
# Don't import tables initially, since it might not be available
# everywhere.
tables = None


from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import control
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from theano import config


logger = logging.getLogger(__name__)


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
    matrix (and optionally, associated targets).


    Parameters
    ----------
    X : ndarray, 2-dimensional, optional
        Should be supplied if `topo_view` is not. A design \
        matrix of shape (number examples, number features) \
        that defines the dataset.
    topo_view : ndarray, optional
        Should be supplied if X is not.  An array whose first \
        dimension is of length number examples. The remaining \
        dimensions are examples with topological significance, \
        e.g. for images the remaining axes are rows, columns, \
        and channels.
    y : ndarray, optional

        Targets for each example (e.g., class ids, values to be predicted
        in a regression task).

        Currently three formats are supported:

        - None:
            Pass `None` if there are no target values. In this case the
            dataset may not be some tasks such as supervised learning
            or evaluation of a supervised learning system, but it can
            be used for some other tasks. For example, a supervised
            learning system can make predictions on it, or an unsupervised
            learning system can be trained on it.
        - 1D ndarray of integers:
            This format may be used when the targets are class labels.
            In this format, the array should have one entry for each
            example. Each entry should be an integer, in the range
            [0, N) where N is the number of classes.
            This is the format that the `SVM` class expects.
        - 2D ndarray, data type optional:
            This is the most common format and can be used for a variety
            of problem types. Each row of the matrix becomes the target
            for a different example. Specific models / costs can interpret
            this target vector differently. For example, the `Linear`
            output layer for the `MLP` class expects the target for each
            example to be a vector of real-valued regression targets. (It
            can be a vector of size one if you only have one regression
            target). The `Softmax` output layer of the `MLP` class expects
            the target to be a vector of N elements, where N is the number
            of classes, and expects all but one of the elements to 0. One
            element should have value 1., and the index of this element
            identifies the target class.
    view_converter : object, optional
        An object for converting between the design matrix \
        stored internally and the topological view of the data.
    rng : object, optional
        A random number generator used for picking random \
        indices into the design matrix when choosing minibatches.
    X_labels : int, optional
        If X contains labels then X_labels must be passed to indicate the
        total number of possible labels e.g. the size of a the vocabulary
        when X contains word indices. This will make the set use
        IndexSpace.
    y_labels : int, optional
        If y contains labels then y_labels must be passed to indicate the
        total number of possible labels e.g. 10 for the MNIST dataset
        where the targets are numbers. This will make the set use
        IndexSpace.

    See Also
    --------
    DenseDesignMatrixPytables : Use this class if your data is too big to fit
        in memory.

    Notes
    -----
    - What kind of data can be stored in this way?
        A design matrix is a matrix where each row contains a single
        example. Each column within the row is a feature of that example.

        By dense, we mean that every entry in the matrix is explicitly given a
        value.

        Examples of datasets that can be stored this way include MNIST and
        CIFAR10.

        Some datasets cannot be stored as a design matrix. For example, a
        collection of images, each image having a different size, can't be
        stored in this way, because we can't reshape each image to the same
        length of matrix row.

        Some datasets can, conceptually, be represented as a design matrix, but
        it may not be efficient to store them as dense matrices. For example,
        a dataset of sentences with a bag of words representation, might have a
        very high number of features but most of the values are zero, so it
        would be better to store the data as a sparse matrix.

    - What if my examples aren't best thought of as vectors?
        The DenseDesignMatrix class supports two views of the data, the "design
        matrix view" in which each example is just a vector, and the
        "topological view" in which each example is formatted using some kind
        of data structure with meaningful topology. For example, a dataset of
        images can be viewed as a design matrix where each row contains a
        flattened version of each image, or it can be viewed as a 4D tensor,
        where each example is a 3D subtensor, with one axis corresponding to
        rows of the image, one axis corresponding to columns of the image, and
        one axis corresponding to the color channels. This structure can be
        thought of as having meaningful topology because neighboring
        coordinates on the row and column axes correspond to neighboring
        pixels in the image.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, axes=('b', 0, 1, 'c'),
                 rng=_default_seed, preprocessor=None, fit_preprocessor=False,
                 max_labels=None, X_labels=None, y_labels=None):
        self.X = X
        self.y = y
        self.view_converter = view_converter
        self.X_labels = X_labels
        self.y_labels = y_labels

        if max_labels is not None:
            warnings.warn("The max_labels argument to DenseDesignMatrix is "
                          "deprecated. Use the y_labels argument instead. The "
                          "max_labels argument will be removed on or after "
                          "6 October 2014", stacklevel=2)
            assert y_labels is None
            self.y_labels = max_labels

        self._check_labels()

        if topo_view is not None:
            assert view_converter is None
            self.set_topological_view(topo_view, axes)
        else:
            assert X is not None, ("DenseDesignMatrix needs to be provided "
                                   "with either topo_view, or X")
            if view_converter is not None:

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
            X_source = 'features'
            if X_labels is None:
                X_space = VectorSpace(dim=X.shape[1])
            else:
                if X.ndim == 1:
                    dim = 1
                else:
                    dim = X.shape[-1]
                X_space = IndexSpace(dim=dim, max_labels=X_labels)
            if y is None:
                space = X_space
                source = X_source
            else:
                if y.ndim == 1:
                    dim = 1
                else:
                    dim = y.shape[-1]
                if y_labels is not None:
                    y_space = IndexSpace(dim=dim, max_labels=y_labels)
                else:
                    y_space = VectorSpace(dim=dim)
                y_source = 'targets'

                space = CompositeSpace((X_space, y_space))
                source = (X_source, y_source)
            self.data_specs = (space, source)
            self.X_space = X_space

        self.compress = False
        self.design_loc = None
        self.rng = make_np_rng(rng, which_method="random_integers")
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_targets = False
        self._iter_data_specs = (self.X_space, 'features')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    def _check_labels(self):
        """Sanity checks for X_labels and y_labels."""
        if self.X_labels is not None:
            assert self.X is not None
            assert self.view_converter is None
            assert self.X.ndim <= 2
            assert np.all(self.X < self.X_labels)

        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            assert np.all(self.y < self.y_labels)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError('In DenseDesignMatrix.iterator, both the '
                                 '"data_specs" argument and deprecated '
                                 'arguments "topo" or "targets" were '
                                 'provided.',
                                 (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead.",
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
                if src == 'features' and \
                   getattr(self, 'view_converter', None) is not None:
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(batch,
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
                                     mode(self.X.shape[0],
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def get_data(self):
        """
        Returns all the data, as it is internally stored.
        The definition and format of these data are described in
        `self.get_data_specs()`.

        Returns
        -------
        data : numpy matrix or 2-tuple of matrices
            The data
        """
        if self.y is None:
            return self.X
        else:
            return (self.X, self.y)

    def use_design_loc(self, path):
        """
        Caling this function changes the serialization behavior of the object
        permanently.

        If this function has been called, when the object is serialized, it
        will save the design matrix to `path` as a .npy file rather
        than pickling the design matrix along with the rest of the dataset
        object. This avoids pickle's unfortunate behavior of using 2X the RAM
        when unpickling.

        TODO: Get rid of this logic, use custom array-aware picklers (joblib,
        custom pylearn2 serialization format).

        Parameters
        ----------
        path : str
            The path to save the design matrix to
        """

        if not path.endswith('.npy'):
            raise ValueError("path should end with '.npy'")

        self.design_loc = path

    def get_topo_batch_axis(self):
        """
        The index of the axis of the batches

        Returns
        -------
        axis : int
            The axis of a topological view of this dataset that corresponds
            to indexing over different examples.
        """
        axis = self.view_converter.axes.index('b')
        return axis

    def enable_compression(self):
        """
        If called, when pickled the dataset will be saved using only
        8 bits per element.

        .. todo::

            Not sure this should be implemented as something a base dataset
            does. Perhaps as a mixin that specific datasets (i.e. CIFAR10)
            inherit from.
        """
        self.compress = True

    def __getstate__(self):
        """
        .. todo::

            WRITEME
        """
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
            rval['X'] = np.cast['uint8'](rval['X'])

        if self.design_loc is not None:
            # TODO: Get rid of this logic, use custom array-aware picklers
            # (joblib, custom pylearn2 serialization format).
            np.save(self.design_loc, rval['X'])
            del rval['X']

        return rval

    def __setstate__(self, d):
        """
        .. todo::

            WRITEME
        """
        if d['design_loc'] is not None:
            if control.get_load_data():
                fname = cache.datasetCache.cache_file(d['design_loc'])
                d['X'] = np.load(fname)
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
                self.X = np.cast['float32'](X) * mx / 255. + mn
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
        _mode : WRITEME
        train_size : int
            Number of examples that will be assigned to the training dataset.
        train_prop : float
            Proportion of training dataset split.

        Returns
        -------
        WRITEME
        """

        """
        This function splits the dataset according to the number of
        train_size if defined by the user with respect to the mode provided
        by the user. Otherwise it will use the
        train_prop to divide the dataset into a training and holdout
        validation set. This function returns the training and validation
        dataset.

        Parameters
        -----------
        _mode : WRITEME
        train_size : int
            Number of examples that will be assigned to the training dataset.
        train_prop : float
            Proportion of training dataset split.

        Returns
        -------
        WRITEME
        """
        if train_size != 0:
            size = train_size
        elif train_prop != 0:
            size = np.round(self.get_num_examples() * train_prop)
        else:
            raise ValueError("Initialize either split ratio and split size to "
                             "non-zero value.")
        if size < self.get_num_examples() - size:
            dataset_iter = self.iterator(
                mode=_mode,
                batch_size=(self.get_num_examples() - size))
            valid = dataset_iter.next()
            train = dataset_iter.next()[:(self.get_num_examples()
                                          - valid.shape[0])]
        else:
            dataset_iter = self.iterator(mode=_mode,
                                         batch_size=size)
            train = dataset_iter.next()
            valid = dataset_iter.next()[:(self.get_num_examples()
                                          - train.shape[0])]
        return (train, valid)

    def split_dataset_nfolds(self, nfolds=0):
        """
        This function splits the dataset into to the number of n folds
        given by the user. Returns an array of folds.

        Parameters
        ----------
        nfolds : int, optional
            The number of folds for the  the validation set.

        Returns
        -------
        WRITEME
        """

        folds_iter = self.iterator(mode="sequential", num_batches=nfolds)
        folds = list(folds_iter)
        return folds

    def split_dataset_holdout(self, train_size=0, train_prop=0):
        """
        This function splits the dataset according to the number of
        train_size if defined by the user.

        Otherwise it will use the train_prop to divide the dataset into a
        training and holdout validation set. This function returns the
        training and validation dataset.

        Parameters
        ----------
        train_size : int
            Number of examples that will be assigned to the training
            dataset.
        train_prop : float
            Proportion of dataset split.
        """
        return self._apply_holdout("sequential", train_size, train_prop)

    def bootstrap_nfolds(self, nfolds, rng=None):
        """
        This function splits the dataset using the random_slice and into the
        n folds. Returns the folds.

        Parameters
        ----------
        nfolds : int
            The number of folds for the  dataset.
        rng : WRITEME
            Random number generation class to be used.
        """

        folds_iter = self.iterator(mode="random_slice",
                                   num_batches=nfolds,
                                   rng=rng)
        folds = list(folds_iter)
        return folds

    def bootstrap_holdout(self, train_size=0, train_prop=0, rng=None):
        """
        This function splits the dataset according to the number of
        train_size defined by the user.

        Parameters
        ----------
        train_size : int
            Number of examples that will be assigned to the training dataset.
        nfolds : int
            The number of folds for the  the validation set.
        rng : WRITEME
            Random number generation class to be used.
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
        .. todo::

            WRITEME properly

        Return to a state specified by an object returned from
        get_stream_position.

        Parameters
        ----------
        pos : object
            WRITEME
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
            self.default_rng = make_np_rng(None, [17, 2, 946],
                                           which_method="random_integers")
        self.rng = copy.copy(self.default_rng)

    def apply_preprocessor(self, preprocessor, can_fit=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        preprocessor : object
            preprocessor object
        can_fit : bool, optional
            WRITEME
        """
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
            An array containing a design matrix representation of
            training examples.

        dspace : Space
            A Space we want the data in mat to be formatted in.
            It can be a VectorSpace for a design matrix output,
            a Conv2DSpace for a topological output for instance.
            Valid values depend on the type of `self.view_converter`.

        Returns
        -------
        WRITEME
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_formatted_view on a dataset "
                            "that has no view converter")

        self.X_space.np_validate(mat)
        return self.view_converter.get_formatted_batch(mat, dspace)

    def get_weights_view(self, mat):
        """
        .. todo::

            WRITEME properly

        Return a view of mat in the topology preserving format. Currently
        the same as get_topological_view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional
            WRITEME
        """

        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        .. todo::

            Why is this parameter named 'V'?

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of
            training examples.
        axes : WRITEME
        """
        assert not contains_nan(V)
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels],
                                                   axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not contains_nan(self.X)

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            # This is to support old pickled models
            if getattr(self, 'y_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            elif getattr(self, 'max_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.max_labels)
            else:
                y_space = VectorSpace(dim=dim)
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

        Returns
        -------
        WRITEME
        """
        if topo is not None:
            if self.view_converter is None:
                raise Exception("Tried to convert from topological_view to "
                                "design matrix using a dataset that has no "
                                "view converter")
            return self.view_converter.topo_view_to_design_mat(topo)

        return self.X

    def set_design_matrix(self, X):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : ndarray
            WRITEME
        """
        assert len(X.shape) == 2
        assert not contains_nan(X)
        self.X = X

    def get_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """

        warnings.warn("num_examples() is being deprecated, and will be "
                      "removed around November 7th, 2014. `get_num_examples` "
                      "should be used instead.",
                      stacklevel=2)

        return self.get_num_examples()

    def get_batch_design(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : int
            WRITEME
        include_labels : bool
            WRITEME
        """
        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                reraise_as(ValueError("Requested %d examples from a dataset "
                                      "containing only %d." %
                                      (batch_size, self.X.shape[0])))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        rx = np.cast[config.floatX](rx)
        return rx

    def get_batch_topo(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : int
            WRITEME
        include_labels : bool
            WRITEME
        """

        if include_labels:
            batch_design, labels = self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels

        return rval

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.X.shape[0]

    def view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.view_converter.view_shape()

    def weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.view_converter.weights_view_shape()

    def has_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y is not None

    def restrict(self, start, stop):
        """
        .. todo::

            WRITEME properly

        Restricts the dataset to include only the examples
        in range(start, stop). Ignored if both arguments are None.

        Parameters
        ----------
        start : int
            start index
        stop : int
            stop index
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
        .. todo::

            WRITEME properly

        If y exists and is a vector of ints, converts it to a binary matrix
        Otherwise will raise some exception

        Parameters
        ----------
        min_class : int
            WRITEME
        """

        if self.y is None:
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix with no labels.")

        if self.y.ndim != 1:
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix whose labels aren't scalar.")

        if 'int' not in str(self.y.dtype):
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix whose labels aren't "
                             "integer-valued.")

        self.y = self.y - min_class

        if self.y.min() < 0:
            raise ValueError("We do not support negative classes. You can use "
                             "the min_class argument to remap negative "
                             "classes to positive values, but we require this "
                             "to be done explicitly so you are aware of the "
                             "remapping.")
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
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : ndarray
            The data to be adjusted
        """
        return X / np.abs(X).max()

    def adjust_to_be_viewed_with(self, X, ref, per_example=None):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : int
            WRITEME
        ref : float
            WRITEME
        per_example : obejct, optional
            WRITEME
        """
        if per_example is not None:
            logger.warning("ignoring per_example")
        return np.clip(X / np.abs(ref).max(), -1., 1.)

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def set_view_converter_axes(self, axes):
        """
        .. todo::

            WRITEME properly

        Change the axes of the view_converter, if any.

        This function is only useful if you intend to call self.iterator
        without data_specs, and with "topo=True", which is deprecated.

        Parameters
        ----------
        axes : WRITEME
            WRITEME
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

    Parameters
    ----------
    X : ndarray, 2-dimensional, optional
        Should be supplied if `topo_view` is not. A design matrix of shape
        (number examples, number features) that defines the dataset.
    topo_view : ndarray, optional
        Should be supplied if X is not.  An array whose first dimension is of
        length number examples. The remaining dimensions are xamples with
        topological significance, e.g. for images the remaining axes are rows,
        columns, and channels.
    y : ndarray, 1-dimensional(?), optional
        Labels or targets for each example. The semantics here are not quite
        nailed down for this yet.
    view_converter : object, optional
        An object for converting between design matrices and topological views.
        Currently DefaultViewConverter is the only type available but later we
        may want to add one that uses the retina encoding that the U of T group
        uses.
    axes : WRITEME
        WRITEME
    rng : object, optional
        A random number generator used for picking random indices into the
        design matrix when choosing minibatches.
    """

    _default_seed = (17, 2, 946)

    def __init__(self,
                 X=None,
                 topo_view=None,
                 y=None,
                 view_converter=None,
                 axes=('b', 0, 1, 'c'),
                 rng=_default_seed):
        super_self = super(DenseDesignMatrixPyTables, self)
        super_self.__init__(X=X,
                            topo_view=topo_view,
                            y=y,
                            view_converter=view_converter,
                            axes=axes,
                            rng=rng)
        ensure_tables()
        if not hasattr(self, 'filters'):
            self.filters = tables.Filters(complib='blosc', complevel=5)

    def set_design_matrix(self, X, start=0):
        """
        .. todo::

            WRITEME
        """
        assert len(X.shape) == 2
        assert not contains_nan(X)
        DenseDesignMatrixPyTables.fill_hdf5(file_handle=self.h5file,
                                            data_x=X,
                                            start=start)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c'), start=0):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        .. todo::

            Why is this parameter named 'V'?

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training \
            examples. If unspecified, the entire dataset (`self.X`) is used \
            instead.
        axes : WRITEME
            WRITEME
        start : WRITEME
        """
        assert not contains_nan(V)
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels],
                                                   axes=axes)
        X = self.view_converter.topo_view_to_design_mat(V)
        assert not contains_nan(X)
        DenseDesignMatrixPyTables.fill_hdf5(file_handle=self.h5file,
                                            data_x=X,
                                            start=start)

    def init_hdf5(self, path, shapes):
        """
        .. todo::

            WRITEME properly

        Initialize hdf5 file to be used ba dataset
        """

        x_shape, y_shape = shapes
        # make pytables
        ensure_tables()
        h5file = tables.openFile(path, mode="w", title="SVHN Dataset")
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = (tables.Float32Atom() if config.floatX == 'float32'
                else tables.Float64Atom())
        h5file.createCArray(gcolumns, 'X', atom=atom, shape=x_shape,
                            title="Data values", filters=self.filters)
        h5file.createCArray(gcolumns, 'y', atom=atom, shape=y_shape,
                            title="Data targets", filters=self.filters)
        return h5file, gcolumns

    @staticmethod
    def fill_hdf5(file_handle,
                  data_x,
                  data_y=None,
                  node=None,
                  start=0,
                  batch_size=5000):
        """
        .. todo::

            WRITEME properly

        PyTables tends to crash if you write large data on them at once.
        This function write data on file_handle in batches

        start: the start index to write data
        """

        if node is None:
            node = file_handle.getNode('/', 'Data')

        data_size = data_x.shape[0]
        last = np.floor(data_size / float(batch_size)) * batch_size
        for i in xrange(0, data_size, batch_size):
            stop = (i + np.mod(data_size, batch_size) if i >= last
                    else i + batch_size)
            assert len(range(start + i, start + stop)) == len(range(i, stop))
            assert (start + stop) <= (node.X.shape[0])
            node.X[start + i: start + stop, :] = data_x[i:stop, :]
            if data_y is not None:
                node.y[start + i: start + stop, :] = data_y[i:stop, :]

            file_handle.flush()

    def resize(self, h5file, start, stop):
        """
        .. todo::

            WRITEME
        """
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

        atom = (tables.Float32Atom() if config.floatX == 'float32'
                else tables.Float64Atom())
        x = h5file.createCArray(gcolumns,
                                'X',
                                atom=atom,
                                shape=((stop - start, data.X.shape[1])),
                                title="Data values",
                                filters=self.filters)
        y = h5file.createCArray(gcolumns,
                                'y',
                                atom=atom,
                                shape=((stop - start, data.y.shape[1])),
                                title="Data targets",
                                filters=self.filters)
        x[:] = data.X[start:stop]
        y[:] = data.y[start:stop]

        h5file.removeNode('/', "Data", 1)
        h5file.renameNode('/', "Data", "Data_")
        h5file.flush()
        return h5file, gcolumns


class DefaultViewConverter(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    shape : list
      [num_rows, num_cols, channels]
    axes : tuple
      The axis ordering to use in topological views of the data. Must be some
      permutation of ('b', 0, 1, 'c'). Default: ('b', 0, 1, 'c')
    """
    def __init__(self, shape, axes=('b', 0, 1, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes
        self._update_topo_space()

    def view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def design_mat_to_topo_view(self, design_matrix):
        """
        Returns a topological view/copy of design matrix.

        Parameters
        ----------
        design_matrix: numpy.ndarray
          A design matrix with data in rows. Data is assumed to be laid out in
          memory according to the axis order ('b', 'c', 0, 1)

        returns: numpy.ndarray
          A matrix with axis order given by self.axes and batch shape given by
          self.shape (if you reordered self.shape to match self.axes, as
          self.shape is always in 'c', 0, 1 order).

          This will try to return
          a view into design_matrix if possible; otherwise it will allocate a
          new ndarray.
        """
        if len(design_matrix.shape) != 2:
            raise ValueError("design_matrix must have 2 dimensions, but shape "
                             "was %s." % str(design_matrix.shape))

        expected_row_size = np.prod(self.shape)
        if design_matrix.shape[1] != expected_row_size:
            raise ValueError("This DefaultViewConverter's self.shape = %s, "
                             "for a total size of %d, but the design_matrix's "
                             "row size was different (%d)." %
                             (str(self.shape),
                              expected_row_size,
                              design_matrix.shape[1]))

        bc01_shape = tuple([design_matrix.shape[0], ] +  # num. batches
                           # Maps the (0, 1, 'c') of self.shape to ('c', 0, 1)
                           [self.shape[i] for i in (2, 0, 1)])
        topo_array_bc01 = design_matrix.reshape(bc01_shape)
        axis_order = [('b', 'c', 0, 1).index(axis) for axis in self.axes]
        return topo_array_bc01.transpose(*axis_order)

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        rval = self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
                                        for axis in ('b', 0, 1, 'c')))

        return rval

    def topo_view_to_design_mat(self, topo_array):
        """
        Returns a design matrix view/copy of topological matrix.

        Parameters
        ----------
        topo_array: numpy.ndarray
          An N-D array with axis order given by self.axes. Non-batch axes'
          dimension sizes must agree with corresponding sizes in self.shape.

        returns: numpy.ndarray
          A design matrix with data in rows. Data, is laid out in memory
          according to the default axis order ('b', 'c', 0, 1). This will
          try to return a view into topo_array if possible; otherwise it will
          allocate a new ndarray.
        """
        for shape_elem, axis in safe_zip(self.shape, (0, 1, 'c')):
            if topo_array.shape[self.axes.index(axis)] != shape_elem:
                raise ValueError(
                    "topo_array's %s axis has a different size "
                    "(%d) from the corresponding size (%d) in "
                    "self.shape.\n"
                    "  self.shape:       %s (uses standard axis order: 0, 1, "
                    "'c')\n"
                    "  self.axes:        %s\n"
                    "  topo_array.shape: %s (should be in self.axes' order)")

        topo_array_bc01 = topo_array.transpose([self.axes.index(ax)
                                                for ax in ('b', 'c', 0, 1)])

        return topo_array_bc01.reshape((topo_array_bc01.shape[0],
                                        np.prod(topo_array_bc01.shape[1:])))

    def get_formatted_batch(self, batch, dspace):
        """
        .. todo::

            WRITEME properly

        Reformat batch from the internal storage format into dspace.
        """
        if isinstance(dspace, VectorSpace):
            # If a VectorSpace is requested, batch should already be in that
            # space. We call np_format_as anyway, in case the batch needs to be
            # cast to dspace.dtype. This also validates the batch shape, to
            # check that it's a valid batch in dspace.
            return dspace.np_format_as(batch, dspace)
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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        self.axes = axes
        self._update_topo_space()


def from_dataset(dataset, num_examples):
    """
    Constructs a random subset of a DenseDesignMatrix

    Parameters
    ----------
    dataset : DenseDesignMatrix
    num_examples : int

    Returns
    -------
    sub_dataset : DenseDesignMatrix
        A new dataset containing `num_examples` examples randomly
        drawn (without replacement) from `dataset`
    """
    try:

        V, y = dataset.get_batch_topo(num_examples, True)

    except TypeError:

        # This patches a case where control.get_load_data() is false so
        # dataset.X is None This logic should be removed whenever we implement
        # lazy loading

        if isinstance(dataset, DenseDesignMatrix) and \
           dataset.X is None and \
           not control.get_load_data():
            warnings.warn("from_dataset wasn't able to make subset of "
                          "dataset, using the whole thing")
            return DenseDesignMatrix(X=None,
                                     view_converter=dataset.view_converter)
        raise

    rval = DenseDesignMatrix(topo_view=V, y=y, y_labels=dataset.y_labels)
    rval.adjust_for_viewer = dataset.adjust_for_viewer

    return rval


def dataset_range(dataset, start, stop):
    """
    Returns a new dataset formed by extracting a range of examples from an
    existing dataset.

    Parameters
    ----------
    dataset : DenseDesignMatrix
        The existing dataset to extract examples from.
    start : int
        Extract examples starting at this index.
    stop : int
        Stop extracting examples at this index. Do not include this index
        itself (like the python `range` builtin)

    Returns
    -------
    sub_dataset : DenseDesignMatrix
        The new dataset containing examples [start, stop).
    """

    if dataset.X is None:
        return DenseDesignMatrix(X=None,
                                 y=None,
                                 view_converter=dataset.view_converter)
    X = dataset.X[start:stop, :].copy()
    if dataset.y is None:
        y = None
    else:
        if dataset.y.ndim == 2:
            y = dataset.y[start:stop, :].copy()
        else:
            y = dataset.y[start:stop].copy()
        assert X.shape[0] == y.shape[0]
    assert X.shape[0] == stop - start
    topo = dataset.get_topological_view(X)
    rval = DenseDesignMatrix(topo_view=topo, y=y)
    rval.adjust_for_viewer = dataset.adjust_for_viewer
    return rval


def convert_to_one_hot(dataset, min_class=0):
    """
    .. todo::

        WRITEME properly

    Convenient way of accessing convert_to_one_hot from a yaml file
    """
    dataset.convert_to_one_hot(min_class=min_class)
    return dataset


def set_axes(dataset, axes):
    """
    .. todo::

        WRITEME
    """
    dataset.set_view_converter_axes(axes)
    return dataset
