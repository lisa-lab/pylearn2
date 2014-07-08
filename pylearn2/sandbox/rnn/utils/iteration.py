from functools import wraps

import numpy as np

from pylearn2.utils import iteration, safe_izip


class ShuffledSequentialSubsetIterator(
        iteration.ShuffledSequentialSubsetIterator):
    fancy = True
    stochastic = True
    uniform_batch_size = False

    def __init__(self, dataset_size, batch_size, num_batches, rng=None,
                 sequence_lengths=None):
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size, batch_size, num_batches, rng
        )
        # If this iterator was called by a dataset with sequences we must
        # make sure that the batches contain sequences of the same length
        if sequence_lengths:
            assert len(sequence_lengths) == dataset_size
            self._sequence_lengths = np.asarray(sequence_lengths)
            self._shuffled_sequence_lengths = \
                self._sequence_lengths[self._shuffled]
            self._create_batches()

    def _create_batches(self):
        """
        This method creates batches given a list of sequence lengths.
        It ensures that batches all have the same sequence lengths.
        """
        # TODO This needs to be optimized a lot; very slow right now
        seen = set()
        batches = []
        while len(seen) < self._dataset_size:
            batch = []
            batch_sequence_length = None
            for index, sequence_length \
                    in safe_izip(self._shuffled,
                                 self._shuffled_sequence_lengths):
                if index not in seen:
                    if not batch_sequence_length:
                        batch_sequence_length = sequence_length
                    if sequence_length == batch_sequence_length:
                        batch.append(index)
                        seen.add(index)
                        if len(batch) == self._batch_size:
                            break
            batches.append(batch)
        self._batches = batches

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
