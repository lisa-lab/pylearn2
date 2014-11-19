"""
Iterator for RNN data
"""
from functools import wraps

import numpy as np
from theano import config

from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.sandbox.rnn.space import SequenceMaskSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_izip
from pylearn2.utils.iteration import FiniteDatasetIterator


class SequenceDatasetIterator(FiniteDatasetIterator):
    """
    Assumes space is a CompositeSpace and source is a tuple.

    Parameters
    ----------
    dataset : `Dataset` object
        The dataset over which to iterate.
    data_specs : tuple
        A `(space, source)` tuple. See :ref:`data_specs` for a full
        description. Must not contain nested composite spaces.
    subset_iterator : object
        An iterator object that returns slice objects or lists of
        examples, conforming to the interface specified by
        :py:class:`SubsetIterator`.
    return_tuple : bool, optional
        Always return a tuple, even if there is exactly one source
        of data being returned. Defaults to `False`.
    convert : list of callables
        A list of callables, in the same order as the sources
        in `data_specs`, that will be called on the individual
        source batches prior to any further processing.

    Notes
    -----
    See the documentation for :py:class:`SubsetIterator` for
    attribute documentation.
    """
    def __init__(self, dataset, data_specs, subset_iterator,
                 return_tuple=False, convert=None):
        # Unpack the data specs into two tuples
        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)

        # Remove the requested mask from the data specs before calling
        # the parent constructor
        self._original_source = source
        mask_seen, sequence_seen = False, False
        self.mask_needed = []
        retain = []
        for i, (subspace, subsource) in enumerate(safe_izip(space.components,
                                                            source)):
            if isinstance(subspace, SequenceMaskSpace):
                if not subsource.endswith('_mask') or \
                        subsource[:-5] not in source:
                    raise ValueError("SequenceDatasetIterator received "
                                     "data_specs containing a "
                                     "SequenceMaskSpace with corresponding "
                                     "source %s, but the source should end "
                                     "with `_mask` in order to match it to the"
                                     "correct SequenceDataSpace")
                mask_seen = True
                self.mask_needed.append(subsource[:-5])
            else:
                retain.append(i)
                if isinstance(subspace, SequenceDataSpace):
                    sequence_seen = True
        if mask_seen != sequence_seen and i + 1 != len(retain):
            raise ValueError("SequenceDatasetIterator was asked to iterate "
                             "over a sequence mask without data or vice versa")
        space = space.restrict(retain)
        source = tuple(source[i] for i in retain)
        super(SequenceDatasetIterator, self).__init__(
            dataset, subset_iterator, (space, source),
            return_tuple=return_tuple, convert=convert
        )
        if not isinstance(space, CompositeSpace):
            space = (space,)
        else:
            space = space.components
        assert len(space) == len(source)
        self._original_space = space

    def __iter__(self):
        return self

    def _create_mask(self, data):
        """
        Creates the mask for a given set of data.

        Parameters
        ----------
        data : numpy sequence of ndarrays
            A sequence of ndarrays representing sequential data
        """
        sequence_lengths = [len(sample) for sample in data]
        max_sequence_length = max(sequence_lengths)
        mask = np.zeros((max_sequence_length, len(data)), dtype=config.floatX)
        for i, sequence_length in enumerate(sequence_lengths):
            mask[:sequence_length, i] = 1
        return mask.T

    @wraps(FiniteDatasetIterator.next)
    def next(self):
        next_index = self._subset_iterator.next()
        rvals = []
        for space, source, data, fn in safe_izip(self._space, self._source,
                                                 self._raw_data,
                                                 self._convert):
            rval = data[next_index]
            if isinstance(space, SequenceDataSpace):
                # Add padding
                max_sequence_length = max(len(sample) for sample
                                          in rval)
                batch = np.zeros((len(rval), max_sequence_length) +
                                 data[0].shape[1:], dtype=data[0].dtype)
                for i, sample in enumerate(rval):
                    batch[i, :len(sample)] = sample
                rval = np.transpose(batch, (1, 0, 2))
                if fn:
                    rval = fn(rval)
                rvals.append(rval)

                # Create mask
                if source in self.mask_needed:
                    rvals.append(self._create_mask(rval))
            else:
                if fn:
                    rval = fn(rval)
                rvals.append(rval)

        # Reorder according to given data specs

        if not self._return_tuple and len(rval) == 1:
            rvals, = rvals
        return tuple(rvals)
