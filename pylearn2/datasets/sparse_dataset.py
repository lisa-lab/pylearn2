from pylearn2.datasets.dataset import Dataset
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
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

class SparseDataset(Dataset):
    """
    SparseDataset is by itself an iterator.
    """
    def __init__(self, load_path=None, from_scipy_sparse_dataset=None, zipped_npy=True):

        self.load_path = load_path
        self.y = None

        if self.load_path != None:
            if zipped_npy == True:
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

        X_space = VectorSpace(dim=self.X.shape[1])
        self.X_space = X_space
        space = self.X_space
        source = 'features'
        self._iter_data_specs = (space, source)
        self.data_specs = (space, source)


    def get_design_matrix(self):
        return self.X_space

    def get_batch_design(self, batch_size, include_labels=False):
        """
        method inherited from Dataset
        """
        self.iterator(mode='sequential', batch_size=batch_size, num_batches=None, topo=None)
        return self.next()

    def get_batch_topo(self, batch_size):
        """
        method inherited from Dataset
        """
        raise NotImplementedError('Not implemented for sparse dataset')

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead. Here these two arguments are not used",
                          stacklevel=2)

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
        return self

    def next(self):
        indx = self.subset_iterator.next()
        try:
            mini_batch = self.X[indx]
        except IndexError:
            # the ind of minibatch goes beyond the boundary
            import ipdb; ipdb.set_trace()
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
