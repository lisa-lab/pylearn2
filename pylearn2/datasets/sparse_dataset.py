from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import SequentialSubsetIterator
import numpy
import warnings
try:
    import scipy.sparse
except ImportError:
    warnings.warn("Couldn't import scipy.sparse")
import theano
import gzip
floatX = theano.config.floatX

class SparseDataset(Dataset):
    """
    SparseDataset is by itself an iterator.
    """
    def __init__(self, load_path=None, from_scipy_sparse_dataset=None, zipped_npy=True):

        self.load_path = load_path

        if self.load_path != None:
            if zipped_npy == True:
                print '... loading sparse data set from a zip npy file'
                self.sparse_matrix = scipy.sparse.csr_matrix(
                    numpy.load(gzip.open(load_path)), dtype=floatX)
            else:
                print '... loading sparse data set from a npy file'
                self.sparse_matrix = scipy.sparse.csr_matrix(
                    numpy.load(load_path).item(), dtype=floatX)
        else:
            print '... building from given sparse dataset'
            self.sparse_matrix = from_scipy_sparse_dataset

        self.data_n_rows = self.sparse_matrix.shape[0]
        self.num_examples = self.data_n_rows

    def get_design_matrix(self):
        return self.sparse_matrix

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
                 topo=None, targets=None, rng=None):
        """
        method inherited from Dataset
        """
        self.mode = mode
        self.batch_size = batch_size
        self._targets = targets

        if mode == 'sequential':
            self.subset_iterator = SequentialSubsetIterator(self.data_n_rows,
                                            batch_size, num_batches, rng=None)
            return self
        else:
            raise NotImplementedError('other iteration scheme not supported for now!')


    def __iter__(self):
        return self

    def next(self):
        indx = self.subset_iterator.next()
        try:
            mini_batch = self.sparse_matrix[indx]
        except IndexError:
            # the ind of minibatch goes beyond the boundary
            import ipdb; ipdb.set_trace()
        return mini_batch
