"""
Iterators providing indices for different kinds of iteration over
datasets.

Presets:
    sequential: iterates through fixed slices of the dataset in sequence
    shuffled_sequential: iterates through a shuffled version of the dataset
                 in sequence
    random_slice: on each call to next, returns a slice of the dataset,
                  chosen uniformly at random over contiguous slices
                  samples with replacement, but still reports that
                  container is empty after num_examples / batch_size calls
    random_uniform: on each call to next, returns a random subset of the
                  dataset.
                  samples with replacement, but still reports that
                  container is empty after num_examples / batch_size calls
"""
from __future__ import division
import warnings
import numpy
np = numpy
from theano import config

from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import is_flat_specs


class SubsetIterator(object):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        Parameters
        ----------
        dataset_size : WRITEME
        batch_size : WRITEME
        num_batches : WRITEME
        rng : int or numpy RandomState
            WRITEME
        """
        raise NotImplementedError()

    def next(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self

    # Class-level attributes that might hint the behaviour of
    # FiniteDatasetIterator.

    # Does this return subsets that need fancy indexing? (i.e. lists
    # of indices)
    fancy = False

    # Does this class make use of random number generators?
    stochastic = False

    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return self._batch_size

    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME
        """
        return self._num_batches

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        return self.batch_size * self.num_batches

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return False


class SequentialSubsetIterator(SubsetIterator):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        .. todo::

            WRITEME
        """
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        assert num_batches is None or num_batches >= 0
        self._dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = int(numpy.ceil(self._dataset_size / num_batches))
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = numpy.ceil(self._dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = numpy.ceil(self._dataset_size / batch_size)
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0
        self._idx = 0
        self._batch = 0

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            self._last = slice(self._idx, self._dataset_size)
            self._idx = self._dataset_size
            return self._last

        else:
            self._last = slice(self._idx, self._idx + self._batch_size)
            self._idx += self._batch_size
            self._batch += 1
            return self._last

    fancy = False
    stochastic = False

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        product = self.batch_size * self.num_batches
        return min(product, self._dataset_size)

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return self.batch_size * self.num_batches > self._dataset_size


class ShuffledSequentialSubsetIterator(SequentialSubsetIterator):
    """
    .. todo::

        WRITEME
    """
    stochastic = True
    fancy = True

    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        .. todo::

            WRITEME
        """
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size,
            batch_size,
            num_batches,
            None
        )
        if rng is not None and hasattr(rng, 'random_integers'):
            self._rng = rng
        else:
            self._rng = numpy.random.RandomState(rng)
        self._shuffled = numpy.arange(self._dataset_size)
        self._rng.shuffle(self._shuffled)

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            rval = self._shuffled[self._idx: self._dataset_size]
            self._idx = self._dataset_size
            return rval
        else:
            rval = self._shuffled[self._idx: self._idx + self._batch_size]
            self._idx += self._batch_size
            self._batch += 1
            return rval


class RandomUniformSubsetIterator(SubsetIterator):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        .. todo::

            WRITEME
        """
        if rng is not None and hasattr(rng, 'random_integers'):
            self._rng = rng
        else:
            self._rng = numpy.random.RandomState(rng)
        if batch_size is None:
            raise ValueError("batch_size cannot be None for random uniform "
                             "iteration")
        elif num_batches is None:
            raise ValueError("num_batches cannot be None for random uniform "
                             "iteration")
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._next_batch_no >= self._num_batches:
            raise StopIteration()
        else:
            self._last = self._rng.random_integers(low=0,
                                                   high=self._dataset_size - 1,
                                                   size=(self._batch_size,))
            self._next_batch_no += 1
            return self._last

    fancy = True
    stochastic = True


class RandomSliceSubsetIterator(RandomUniformSubsetIterator):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        .. todo::

            WRITEME
        """
        if batch_size is None:
            raise ValueError("batch_size cannot be None for random slice "
                             "iteration")
        elif num_batches is None:
            raise ValueError("num_batches cannot be None for random slice "
                             "iteration")
        super(RandomSliceSubsetIterator, self).__init__(dataset_size,
                                                        batch_size,
                                                        num_batches, rng)
        self._last_start = self._dataset_size - self._batch_size
        if self._last_start < 0:
            raise ValueError("batch_size > dataset_size not supported for "
                             "random slice iteration")

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._next_batch_no >= self._num_batches:
            raise StopIteration()
        else:
            start = self._rng.random_integers(low=0, high=self._last_start)
            self._last = slice(start, start + self._batch_size)
            self._next_batch_no += 1
            return self._last

    fancy = False
    stochastic = True

class BatchwiseShuffledSequentialIterator(SequentialSubsetIterator):
    """
    Returns minibatches randomly, but sequential inside each minibatch
    """

    def __init__(self, dataset_size, batch_size, num_batches = None, rng=None):
        """
        .. todo::

            WRITEME
        """
        if rng is not None and hasattr(rng, 'random_integers'):
            self._rng = rng
        else:
            self._rng = numpy.random.RandomState(rng)

        assert num_batches is None or num_batches >= 0
        self._dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = int(numpy.ceil(self._dataset_size / num_batches))
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = numpy.ceil(self._dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = numpy.ceil(self._dataset_size / batch_size)

        self._batch_size = batch_size
        self._num_batches = int(num_batches)
        self._next_batch_no = 0
        self._idx = 0
        self._batch_order = range(self._num_batches)
        self._rng.shuffle(self._batch_order)

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._next_batch_no >= self._num_batches:
            raise StopIteration()
        else:
            start = self._batch_order[self._next_batch_no] * self._batch_size
            if start + self._batch_size > self._dataset_size:
                self._last = slice(start, self._dataset_size)
            else:
                self._last = slice(start, start + self._batch_size)
            self._next_batch_no += 1
            return self._last

    fancy = False
    stochastic = True


_iteration_schemes = {
    'sequential': SequentialSubsetIterator,
    'shuffled_sequential': ShuffledSequentialSubsetIterator,
    'random_slice': RandomSliceSubsetIterator,
    'random_uniform': RandomUniformSubsetIterator,
    'batchwise_shuffled_sequential': BatchwiseShuffledSequentialIterator,
}


def is_stochastic(mode):
    """
    .. todo::

        WRITEME
    """
    return resolve_iterator_class(mode).stochastic

def resolve_iterator_class(mode):
    """
    .. todo::

        WRITEME
    """
    if isinstance(mode, basestring) and mode not in _iteration_schemes:
        raise ValueError("unknown iteration mode string: %s" % mode)
    elif mode in _iteration_schemes:
        subset_iter_class = _iteration_schemes[mode]
    else:
        subset_iter_class = mode
    return subset_iter_class


class FiniteDatasetIterator(object):
    """
    A thin wrapper around one of the mode iterators.
    """
    def __init__(self, dataset, subset_iterator, topo=None, targets=None,
                 data_specs=None, return_tuple=False, convert=None):
        """
        .. todo::

            WRITEME
        """
        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError("In FiniteDatasetIterator, both "
                        "the `data_specs` argument and deprecated arguments "
                        "`topo` or `targets` were provided.",
                        (data_specs, topo, targets))

            if convert is not None:
                raise ValueError("In %s, both the `convert` argument and "
                        "deprecated arguments `topo` or `targets` were "
                        "provided." % self.__class__.__name__,
                        (convert, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are being "
                    "deprecated, and will be removed around November 7th, "
                    "2013. `data_specs` should be used instead.",
                    stacklevel=2)
            topo = False
            targets = False
            self._deprecated_interface = True
        else:
            self._deprecated_interface = False

        self._topo = topo
        self._targets = targets
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        # TODO: More thought about how to handle things where this
        # fails (gigantic HDF5 files, etc.)
        if self._deprecated_interface:
            self._raw_data = self._dataset.get_design_matrix()
            if self._targets:
                self._raw_targets = self._dataset.get_targets()
                if self._raw_targets is None:
                    raise ValueError("Can't iterate with targets=True on a "
                                     "dataset object with no targets")
                self._targets_need_cast = not np.dtype(config.floatX) == self._raw_targets.dtype
            self._needs_cast = not np.dtype(config.floatX) == self._raw_data.dtype
        else:
            # Keep only the needed sources in self._raw_data.
            # Remember what source they correspond to in self._source
            assert is_flat_specs(data_specs)

            dataset_space, dataset_source = self._dataset.get_data_specs()
            assert is_flat_specs((dataset_space, dataset_source))

            # the dataset's data spec is either a single (space, source) pair,
            # or a pair of (non-nested CompositeSpace, non-nested tuple).
            # We could build a mapping and call flatten(..., return_tuple=True)
            # but simply putting spaces, sources and data in tuples is simpler.
            if not isinstance(dataset_source, tuple):
                dataset_source = (dataset_source,)

            if not isinstance(dataset_space, CompositeSpace):
                dataset_sub_spaces = (dataset_space,)
            else:
                dataset_sub_spaces = dataset_space.components
            assert len(dataset_source) == len(dataset_sub_spaces)

            all_data = self._dataset.get_data()
            if not isinstance(all_data, tuple):
                all_data = (all_data,)

            space, source = data_specs
            if not isinstance(source, tuple):
                source = (source,)
            if not isinstance(space, CompositeSpace):
                sub_spaces = (space,)
            else:
                sub_spaces = space.components
            assert len(source) == len(sub_spaces)

            self._raw_data = tuple(all_data[dataset_source.index(s)]
                                   for s in source)
            self._source = source

            if convert is None:
                self._convert = [None for s in source]
            else:
                assert len(convert) == len(source)
                self._convert = convert

            for i, (so, sp) in enumerate(safe_zip(source, sub_spaces)):
                idx = dataset_source.index(so)
                dspace = dataset_sub_spaces[idx]

                init_fn = self._convert[i]
                fn = init_fn
                # Compose the functions
                needs_cast = not (np.dtype(config.floatX) ==
                                  self._raw_data[i].dtype)
                if needs_cast:
                    if fn is None:
                        fn = lambda batch: numpy.cast[config.floatX](batch)
                    else:
                        fn = (lambda batch, fn_=fn:
                              numpy.cast[config.floatX](fn_(batch)))

                # If there is an init_fn, it is supposed to take
                # care of the formatting, and it should be an error
                # if it does not. If there was no init_fn, then
                # the iterator will try to format using the generic
                # space-formatting functions.
                needs_format = not init_fn and not sp == dspace
                if needs_format:
                    # "dspace" and "sp" have to be passed as parameters
                    # to lambda, in order to capture their current value,
                    # otherwise they would change in the next iteration
                    # of the loop.
                    if fn is None:
                        fn = (lambda batch, dspace=dspace, sp=sp:
                              dspace.np_format_as(batch, sp))
                    else:
                        fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                              dspace.np_format_as(fn_(batch), sp))

                self._convert[i] = fn

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
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using numpy.take()

        # This saves us some memory (and time spent allocating it)
        # when the dataset dtype matches floatX and next_index is not a
        # fancy-index.
        if self._deprecated_interface:
            if self._needs_cast:
                features = numpy.cast[config.floatX](self._raw_data[next_index])
            else:
                features = self._raw_data[next_index]
            if self._topo:
                features = self._dataset.get_topological_view(features)
            if self._targets:
                targets = self._raw_targets[next_index]
                if self._targets_need_cast:
                    targets = np.cast[config.floatX](targets)
                return features, targets
            else:
                return features
        else:
            rval = tuple(
                    fn(data[next_index]) if fn else data[next_index]
                    for data, fn in safe_zip(self._raw_data, self._convert))
            if not self._return_tuple and len(rval) == 1:
                rval, = rval
            return rval

    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.batch_size

    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.num_batches

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.num_examples

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.uneven

    @property
    def stochastic(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.stochastic

class FiniteDatasetIteratorPyTables(FiniteDatasetIterator):
    """
    .. todo::

        WRITEME
    """
    def next(self):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("This class is obselete with the new interface change, "
                    "and will be removed around November 7th",
                    stacklevel=2)

        next_index = self._subset_iterator.next()

        if self._deprecated_interface:
            if isinstance(next_index, np.ndarray) and len(next_index) == 1:
                next_index = next_index[0]
            if self._needs_cast:
                features = numpy.cast[config.floatX](self._raw_data[next_index])
            else:
                features = self._raw_data[next_index,:]
            if self._topo:
                if len(features.shape) != 2:
                    features = features.reshape((1, features.shape[0]))
                features = self._dataset.get_topological_view(features)
            if self._targets:
                targets = self._raw_targets[next_index,:]
                if len(targets.shape) != 2:
                    targets = targets.reshape((1, targets.shape[0]))
                if self._targets_need_cast:
                    targets = np.cast[config.floatX](targets)
                return features, targets
            else:
                return features
        else:
            rval = tuple(
                    fn(data[next_index]) if fn else data[next_index]
                    for data, fn in safe_zip(self._raw_data, self._convert))
            if not self._return_tuple and len(rval) == 1:
                rval, = rval
            return rval
