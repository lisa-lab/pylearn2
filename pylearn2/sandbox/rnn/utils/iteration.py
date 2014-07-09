"""
Subset iterators which have been adapted for RNNs.
"""
from functools import wraps

import numpy as np

from pylearn2.utils import iteration, safe_izip


class SequentialSubsetIterator(iteration.SequentialSubsetIterator):
    """
    An RNN-friendly version of the SequentialSubsetIterator. If passed
    a list of sequence lengths, it will create batches which only
    contain sequences of the same length.
    Returns mini-batches proceeding sequentially through the dataset.

    See :py:class:`SubsetIterator` for detailed constructor parameter
    and attribute documentation.

    Parameters
    ----------
    sequence_lengths : list or array of ints
        This should contain a list of sequence lengths in the
        same order as the dataset, allowing the iterator to return
        batches with uniform sequence lengths.
    """
    def __init__(self, dataset_size, batch_size, num_batches, rng=None,
                 sequence_lengths=None):
        super(SequentialSubsetIterator, self).__init__(
            dataset_size, batch_size, num_batches, rng
        )
        # If this iterator was called by a dataset with sequences we must
        # make sure that the batches contain sequences of the same length
        if sequence_lengths:
            assert len(sequence_lengths) == dataset_size
            self._sequence_lengths = np.asarray(sequence_lengths)
            self._indices = np.arange(self._dataset_size)
            self._create_batches(self._indices, self._sequence_lengths)

    def _create_batches(self, indices, sequence_lengths):
        """
        This method creates batches with the same sequence lengths.

        Parameters
        ----------
        indices : list of ints or numpy array
            A list of indices; can be either shuffled or simply
            sequential, but must be in the same order as the
            sequence lengths
        sequence_lengths : list of ints or numpy array
            The lengths of the sequences in the same order as indices.
        """
        # TODO This needs to be optimized a lot; very slow right now
        seen = set()
        batches = []
        while len(seen) < self._dataset_size:
            batch = []
            batch_sequence_length = None
            for index, sequence_length \
                    in safe_izip(indices, sequence_lengths):
                if index not in seen:
                    if not batch_sequence_length:
                        batch_sequence_length = sequence_length
                    if sequence_length == batch_sequence_length:
                        batch.append(index)
                        if len(batch) == self._batch_size:
                            break
            seen.update(batch)
            batches.append(batch)
        self._batches = batches

    @wraps(iteration.SequentialSubsetIterator.next)
    def next(self):
        # Either use the pre-created batches with equal sequence lengths,
        # or else just call the base class's default next method
        if hasattr(self, '_sequence_lengths'):
            if self._batch >= len(self._batches):
                raise StopIteration()
            else:
                self._last = self._batches[self._batch]
                self._idx += len(self._last)
                self._batch += 1
                return self._last
        else:
            return super(SequentialSubsetIterator, self).next()


class ShuffledSequentialSubsetIterator(
        iteration.ShuffledSequentialSubsetIterator, SequentialSubsetIterator):
    """
    Randomly shuffles the example indices and then proceeds sequentially
    through the permutation.

    Parameters
    ----------
    See :py:class:`SequentialSubsetIterator` for detailed constructor
    parameter and attribute documentation.
    """
    # Inherit from SequentialSubsetIterator for _create_batches
    def __init__(self, dataset_size, batch_size, num_batches, rng=None,
                 sequence_lengths=None):
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size, batch_size, num_batches, rng
        )
        if sequence_lengths:
            assert len(sequence_lengths) == dataset_size
            self._sequence_lengths = np.asarray(sequence_lengths)
            self._shuffled_sequence_lengths = \
                self._sequence_lengths[self._shuffled]
            self._create_batches(self._shuffled,
                                 self._shuffled_sequence_lengths)

    @wraps(iteration.ShuffledSequentialSubsetIterator.next)
    def next(self):
        # Either use the pre-created batches with equal sequence lengths,
        # or else just call the base class's default next method
        if hasattr(self, '_sequence_lengths'):
            if self._batch >= len(self._batches):
                raise StopIteration()
            else:
                self._last = self._batches[self._batch]
                self._idx += len(self._last)
                self._batch += 1
                return self._last
        else:
            return super(ShuffledSequentialSubsetIterator, self).next()
