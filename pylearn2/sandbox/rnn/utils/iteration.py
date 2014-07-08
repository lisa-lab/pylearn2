from functools import wraps

import numpy as np

from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import iteration, safe_izip
from pylearn2.utils.data_specs import is_flat_specs


class ShuffledSequentialSubsetIterator(
        iteration.ShuffledSequentialSubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None,
                 sequence_lengths=None):
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size, batch_size, num_batches, rng
        )
        if sequence_lengths:
            assert len(sequence_lengths) == dataset_size
            sequence_lengths = np.asarray(sequence_lengths)
            self._create_batches(sequence_lengths)
            self._shuffled_sequence_lengths = sequence_lengths[self._shuffled]

    def _create_batches(self):
        """
        This method creates batches given a list of sequence lengths.
        It ensures that batches all have the same sequence lengths.
        """
        seen = ()
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
        if self.sequence_lengths:
            if self._batch >= len(self._batches):
                raise StopIteration()
            else:
                self._last = self._batches[self._batch]
                self._idx += len(self._last)
                self._batch += 1
                return self._last
        else:
            return super(ShuffledSequentialSubsetIterator, self).next()


class FiniteDatasetIterator(
        iteration.FiniteDatasetIterator):
    @wraps(iteration.FiniteDatasetIterator)
    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

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

        for i, (so, sp, dt) in enumerate(safe_izip(source,
                                                   sub_spaces,
                                                   self._raw_data)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            if init_fn is None:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:

                    def fn(batch, dspace=dspace, sp=sp):
                        if isinstance(dspace, SequenceSpace):
                            # Data is stores as [batch, data, time], and we
                            # want [time, batch, data]
                            batch = np.array([_ for _ in batch])
                            batch = np.transpose(batch, (2, 0, 1))
                        try:
                            return dspace.np_format_as(batch, sp)
                        except ValueError as e:
                            msg = str(e) + ('\nMake sure that the model and '
                                            'dataset have been initialized '
                                            'with correct values.')
                            raise ValueError(msg)
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn
