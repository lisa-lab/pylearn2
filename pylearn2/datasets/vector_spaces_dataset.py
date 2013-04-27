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
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.data_specs import is_flat_specs
from theano import config


def ensure_tables():
    """
    Makes sure tables module has been imported
    """

    global tables
    if tables is None:
        import tables


class VectorSpacesDataset(Dataset):
    """
    A class representing datasets being stored as a number of VectorSpaces.

    This can be seen as a generalization of DenseDesignMatrix where
    there can be any number of sources, not just X and possibly y.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, data=None, data_specs=None, rng=_default_seed,
                 preprocessor=None, fit_preprocessor=False):
        """
        Parameters
        ----------
        data: ndarray, or tuple of ndarrays, containing the data.
            It is formatted as specified in `data_specs`.
            For instance, if `data_specs` is (VectorSpace(nfeat), 'features'),
            then `data` has to be a 2-d ndarray, of shape (nb examples,
            nfeat), that defines an unlabeled dataset. If `data_specs`
            is (CompositeSpace(Conv2DSpace(...), VectorSpace(1)),
            ('features', 'target')), then `data` has to be an (X, y) pair,
            with X being an ndarray containing images stored in the topological
            view specified by the `Conv2DSpace`, and y being a 2-D ndarray
            of width 1, containing the labels or targets for each example.

        data_specs: A (space, source) pair, where space is an instance of
            `Space` (possibly a `CompositeSpace`), and `source` is a
            string (or tuple of strings, if `space` is a `CompositeSpace`),
            defining the format and labels associated to `data`.

        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.

        preprocessor: WRITEME

        fit_preprocessor: WRITEME
        """
        # data_specs should be flat, and there should be no
        # duplicates in source, as we keep only one version
        assert is_flat_specs(data_specs)
        if isinstance(data_specs[1], tuple):
            assert sorted(set(data_specs[1])) == sorted(data_specs[1])
        self.data = data
        self.data_specs = data_specs

        self.compress = False
        self.design_loc = None
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None):

        if topo is not None or targets is not None:
            raise ValueError("You should use the new interface iterator")

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
            data_specs = self.data_specs
        return FiniteDatasetIterator(
                self,
                mode(self.data_specs[0].get_batch_size(self.data),
                     batch_size, num_batches, rng),
                data_specs=data_specs)

    def get_data(self):
        return self.data

    def set_data(self, data, data_specs):
        # data is organized as data_specs
        # keep self.data_specs, and convert data
        data_specs[0].validate(data)
        assert not [N.any(N.isnan(X)) for X in data]
        raise NotImplementedError()

    def get_source(self, name):
        raise NotImplementedError()

    @property
    def num_examples(self):
        return self.data_specs[0].get_batch_size(self.data)

    def get_batch(self, batch_size, data_specs=None):
        raise NotImplementedError()
        """
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
        """
