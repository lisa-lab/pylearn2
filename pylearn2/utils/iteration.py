"""
Iterators providing indices for different kinds of iteration over
datasets.
"""
from __future__ import division
import warnings
import numpy
from theano import config

class SubsetIterator(object):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        raise NotImplementedError()

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    # Class-level attributes that might hint the behaviour of
    # FiniteDatasetIterator.

    # Does this return subsets that need fancy indexing? (i.e. lists
    # of indices)
    fancy = False

    # Does this class make use of random number generators?
    stochastic = False


class SequentialSubsetIterator(SubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        self.dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = numpy.ceil(self.dataset_size / num_batches)
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                # TODO: some might prefer we just raise an exception here.
                # A warning seems like sensible enough behaviour to me,
                # but this may change.
                warnings.warn("Ignoring num_batches argument in presence "
                              "of batch_size argument for sequential "
                              "iteration")
        self.batch_size = batch_size
        self.current = 0

    def next(self):
        if self.current >= self.dataset_size:
            raise StopIteration()
        else:
            self._last = slice(self.current, self.current + self.batch_size)
            self.current += self.batch_size
            return self._last

    fancy = False
    stochastic = False


class RandomUniformSubsetIterator(SubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
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
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._next_batch_no = 0

    def next(self):
        if self._next_batch_no >= self.num_batches:
            raise StopIteration()
        else:
            self._last = self._rng.random_integers(low=0,
                                                   high=self.dataset_size - 1,
                                                   size=(self.batch_size,))
            self._next_batch_no += 1
            return self._last

    fancy = True
    stochastic = True


class RandomSliceSubsetIterator(RandomUniformSubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        if batch_size is None:
            raise ValueError("batch_size cannot be None for random slice "
                             "iteration")
        elif num_batches is None:
            raise ValueError("num_batches cannot be None for random slice "
                             "iteration")
        super(RandomSliceSubsetIterator, self).__init__(dataset_size,
                                                        batch_size,
                                                        num_batches, rng)
        self._last_start = self.dataset_size - self.batch_size
        if self._last_start < 0:
            raise ValueError("batch_size > dataset_size not supported for "
                             "random slice iteration")

    def next(self):
        if self._next_batch_no >= self.num_batches:
            raise StopIteration()
        else:
            start = self._rng.random_integers(low=0, high=self._last_start)
            self._last = slice(start, start + self.batch_size)
            self._next_batch_no += 1
            return self._last

    fancy = False
    stochastic = True


_iteration_schemes = {
    'sequential': SequentialSubsetIterator,
    'random_slice': RandomSliceSubsetIterator,
    'random_uniform': RandomUniformSubsetIterator,
}


def resolve_iterator_class(mode):
    if isinstance(mode, basestring) and mode not in _iteration_schemes:
        raise ValueError("unknown iteration mode string: %s" % mode)
    elif mode in _iteration_schemes:
        subset_iter_class = _iteration_schemes[mode]
    else:
        subset_iter_class = mode
    return subset_iter_class


class FiniteDatasetIterator(object):
    """A thin wrapper around one of the mode iterators."""
    def __init__(self, dataset, subset_iterator, topo=False, targets=False):
        self._topo = topo
        self._targets = targets
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        # TODO: More thought about how to handle things where this
        # fails (gigantic HDF5 files, etc.)
        if self._topo:
            self._raw_data = self._dataset.get_topological_view()
        else:
            self._raw_data = self._dataset.get_design_matrix()
        if self._targets:
            self._raw_targets = self._dataset.get_targets()
            if self._raw_targets is None:
                raise ValueError("Can't iterate with targets=True on a "
                                 "dataset object with no targets")

    def __iter__(self):
        return self

    def next(self):
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using numpy.take()
        features = numpy.cast[config.floatX](self._raw_data[next_index])
        if self._targets:
            return features, self._raw_targets[next_index]
        else:
            return features
