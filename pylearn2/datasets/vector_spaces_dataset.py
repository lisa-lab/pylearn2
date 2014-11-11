"""TODO: module-level docstring."""
__authors__ = "Pascal Lamblin and Razvan Pascanu"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Pascal Lamblin", "Razvan Pascanu", "Ian Goodfellow",
               "Mehdi Mirza"]
__license__ = "3-clause BSD"
__maintainer__ = "Pascal Lamblin"
__email__ = "lamblinp@iro"
import functools

import numpy as np

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import wraps
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan


class VectorSpacesDataset(Dataset):

    """
    A class representing datasets being stored as a number of VectorSpaces.

    This can be seen as a generalization of DenseDesignMatrix where
    there can be any number of sources, not just X and possibly y.

    Parameters
    ----------
    data : ndarray, or tuple of ndarrays, containing the data.
        It is formatted as specified in `data_specs`. For instance, if
        `data_specs` is (VectorSpace(nfeat), 'features'), then `data` has to be
        a 2-d ndarray, of shape (nb examples, nfeat), that defines an unlabeled
        dataset. If `data_specs` is (CompositeSpace(Conv2DSpace(...),
        VectorSpace(1)), ('features', 'target')), then `data` has to be an
        (X, y) pair, with X being an ndarray containing images stored in the
        topological view specified by the `Conv2DSpace`, and y being a 2-D
        ndarray of width 1, containing the labels or targets for each example.
    data_specs : (space, source) pair
        space is an instance of `Space` (possibly a `CompositeSpace`),
        and `source` is a string (or tuple of strings, if `space` is a
        `CompositeSpace`), defining the format and labels associated
        to `data`.
    rng : object, optional
        A random number generator used for picking random indices into the
        design matrix when choosing minibatches.
    preprocessor: WRITEME
    fit_preprocessor: WRITEME
    """
    _default_seed = (17, 2, 946)

    def __init__(self, data=None, data_specs=None, rng=_default_seed,
                 preprocessor=None, fit_preprocessor=False):
        # data_specs should be flat, and there should be no
        # duplicates in source, as we keep only one version
        assert is_flat_specs(data_specs)
        if isinstance(data_specs[1], tuple):
            assert sorted(set(data_specs[1])) == sorted(data_specs[1])
        space, source = data_specs
        space.np_validate(data)
        assert len(set(elem.shape[0] for elem in list(data))) <= 1
        self.data = data
        self.data_specs = data_specs
        self.num_examples = list(data)[0].shape[0]

        self.compress = False
        self.design_loc = None
        self.rng = make_np_rng(rng, which_method='random_integers')
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):

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
            mode(self.get_num_examples(),
                 batch_size, num_batches, rng),
            data_specs=data_specs, return_tuple=return_tuple
        )

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def get_data(self):
        """
        .. todo::

            WRITEME
        """
        return self.data

    def set_data(self, data, data_specs):
        """
        .. todo::

            WRITEME
        """
        # data is organized as data_specs
        # keep self.data_specs, and convert data
        data_specs[0].np_validate(data)
        assert not [contains_nan(X) for X in data]
        raise NotImplementedError()

    def get_source(self, name):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    @wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.num_examples

    def get_batch(self, batch_size, data_specs=None):
        """
        .. todo::

            WRITEME
        """
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
