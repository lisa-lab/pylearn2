"""
Iterators providing indices for different kinds of iteration over
datasets.
"""
from __future__ import division
import warnings
import numpy


class SequentialSubsetIterator(object):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        self.dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = numpy.ceil(self.X.shape[0] / num_batches)
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

    def __iter__(self):
        return self

    def next(self):
        if self.current >= self.dataset_size:
            raise StopIteration()
        else:
            self._last = slice(self.current, self.current + self.batch_size)
            self.current += self.batch_size
            return self._last

    fancy = False
    stochastic = False


class RandomUniformSubsetIterator(object):
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

    def __iter__(self):
        return self

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
            raise ValueError("num_batches cannot be None for random uniform "
                             "iteration")
        super(RandomSliceSubsetIterator, self).__init__(dataset_size,
                                                        batch_size,
                                                        num_batches, rng)
        self._last_start = self.dataset_size - self.batch_size
        if self._last_start < 0:
            raise ValueError("batch_size > dataset_size not supported for "
                             "random slice iteration")

    def __iter__(self):
        return self

    def next(self):
        if self._next_batch_no >= self.num_batches:
            raise StopIteration()
        else:
            start = self._rng.random_integers(low=0, high=self._last_start)
            self._last = slice(start, start + self.batch_size)
            self._next_batch_no += 1
            return self._last

    def reset(self):
        self._next_batch_no = 0

    fancy = False
    stochastic = True
