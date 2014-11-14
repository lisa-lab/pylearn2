"""
.. todo::

    WRITEME
"""
import functools

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import wraps
import logging
import numpy
import warnings
try:
    import scipy.sparse
except ImportError:
    warnings.warn("Couldn't import scipy.sparse")
import theano
import gzip
floatX = theano.config.floatX
logger = logging.getLogger(__name__)
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)


class SparseDataset(Dataset):

    """
    SparseDataset is a class for representing datasets that can be
    stored as a sparse matrix.

    Parameters
    ----------
    load_path : str or None, optional
        the path to read the sparse dataset
        from_scipy_sparse_dataset is not used if load_path is specified
    from_scipy_sparse_dataset : matrix of type scipy.sparse or None, optional
        In case load_path is not provided,
        the sparse dataset is passed directly to the class by
        using from_scipy_sparse_dataset parameter.
    zipped_npy : bool, optional
        used only when load_path is specified.
        indicates whether the input matrix is zipped or not.
        defaults to True.
    """

    def __init__(self, load_path=None,
                 from_scipy_sparse_dataset=None, zipped_npy=True):

        self.load_path = load_path
        self.y = None

        if self.load_path is not None:
            if zipped_npy is True:
                logger.info('... loading sparse data set from a zip npy file')
                self.X = scipy.sparse.csr_matrix(
                    numpy.load(gzip.open(load_path)), dtype=floatX)
            else:
                logger.info('... loading sparse data set from a npy file')
                self.X = scipy.sparse.csr_matrix(
                    numpy.load(load_path).item(), dtype=floatX)
        else:
            logger.info('... building from given sparse dataset')
            self.X = from_scipy_sparse_dataset
            if not scipy.sparse.issparse(from_scipy_sparse_dataset):
                msg = "from_scipy_sparse_dataset is not sparse : %s" \
                      % type(self.X)
                raise TypeError(msg)

        X_space = VectorSpace(dim=self.X.shape[1], sparse=True)
        self.X_space = X_space
        space = self.X_space
        source = 'features'
        self._iter_data_specs = (space, source)
        self.data_specs = (space, source)

    def get_design_matrix(self):
        """
        .. todo::

            WRITEME
        """
        return self.X

    @wraps(Dataset.get_batch_design)
    def get_batch_design(self, batch_size, include_labels=False):
        """Method inherited from Dataset"""
        self.iterator(mode='shuffled_sequential',
                      batch_size=batch_size, num_batches=None)
        return self.next()

    @wraps(Dataset.get_batch_topo)
    def get_batch_topo(self, batch_size):
        """Method inherited from Dataset"""
        raise NotImplementedError('Not implemented for sparse dataset')

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.X.shape[0]

    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):

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

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self

    def next(self):
        """
        .. todo::

            WRITEME
        """
        indx = self.subset_iterator.next()
        try:
            mini_batch = self.X[indx]
        except IndexError as e:
            reraise_as(ValueError("Index out of range" + str(e)))
            # the ind of minibatch goes beyond the boundary
        return mini_batch

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def get_data(self):
        """
        Returns
        -------
        data : numpy matrix or 2-tuple of matrices
            Returns all the data, as it is internally stored.
            The definition and format of these data are described in
            `self.get_data_specs()`.
        """
        if self.y is None:
            return self.X
        else:
            return (self.X, self.y)
