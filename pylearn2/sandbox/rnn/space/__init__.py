"""
Spaces specific to the RNN framework, specifically the SequenceSpace
"""
from __future__ import print_function
from functools import wraps

import numpy as np
from theano import scan, config
from theano import tensor
from theano.tensor import TensorType

from pylearn2 import space
from pylearn2.space import Conv2DSpace
from pylearn2.utils import is_iterable


class SequenceSpace(space.CompositeSpace):
    """
    This space represents sequences. In order to create batches it
    actually consists of two sub-spaces, representing the data and
    the mask (a binary-valued matrix).

    Parameters
    ----------
    space : Space subclass
        The space of which this is a sequence e.g. for a sequence of
        integers use SequenceDataSpace(IndexSpace(dim=...))
    """
    def __init__(self, space):
        self.space = space
        self.mask_space = SequenceMaskSpace()
        self.data_space = SequenceDataSpace(space)
        self.dim = space.get_total_dimension()
        super(SequenceSpace, self).__init__([self.data_space, self.mask_space])
        self._dtype = self._clean_dtype_arg(space.dtype)

    @wraps(space.Space.__eq__)
    def __eq__(self, other):
        if (not isinstance(other, self.__class__) and
                not issubclass(self.__class__, other)):
            return False
        return self.space == other.space

    @wraps(space.Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        data_batch = self.data_space.make_theano_batch(name=name, dtype=dtype,
                                                       batch_size=batch_size)
        mask_batch = self.mask_space.make_theano_batch(name=name, dtype=dtype,
                                                       batch_size=batch_size)
        return (data_batch, mask_batch)

    @wraps(space.Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return self.data_space._batch_size_impl(is_numeric, batch[0])

    @wraps(space.Space.get_total_dimension)
    def get_total_dimension(self):
        return self.space.get_total_dimension()

    @wraps(space.Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        assert is_iterable(batch) and len(batch) == 2
        self.data_space._validate_impl(is_numeric, batch[0])
        self.mask_space._validate_impl(is_numeric, batch[1])

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, self.space)

    @wraps(space.Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        assert isinstance(space, SequenceSpace)
        if is_numeric:
            rval = np.apply_over_axes(
                lambda batch, axis: self.space._format_as_impl(
                    is_numeric=is_numeric,
                    batch=batch,
                    space=space.space),
                batch, 0)
        else:
            NotImplementedError("Can't convert SequenceSpace Theano variables")
        return rval


class SequenceDataSpace(space.SimplyTypedSpace):
    """
    The data space stores the actual data in the format
    (time, batch, data, ..., data).

    Parameters
    ----------
    space : Space subclass
        The space of which this is a sequence e.g. for a sequence of
        integers use SequenceDataSpace(IndexSpace(dim=...))
    """
    def __init__(self, space):
        self.dim = space.get_total_dimension()
        self.space = space
        self._dtype = self._clean_dtype_arg(space.dtype)
        super(SequenceDataSpace, self).__init__(space.dtype)

    def __eq__(self, other):
        if not isinstance(other, SequenceDataSpace):
            return False
        return self.space == other.space

    def __hash__(self):
        return hash(self.space)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, self.space)

    @wraps(space.Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if space == self:
            return batch
        else:
            if isinstance(space, SequenceDataSpace):
                if is_numeric:
                    formatted_batch = np.transpose(np.asarray([
                        self.space._format_as_impl(is_numeric, sample,
                                                   space.space)
                        for sample in np.transpose(batch, (1, 0, 2))
                    ]), (1, 0, 2))
                else:
                    formatted_batch, _ = scan(
                        fn=lambda elem: self.space._format_as_impl(
                            is_numeric, elem, space.space),
                        sequences=[batch]
                    )
                return formatted_batch
            elif isinstance(space, space.VectorSpace):
                row = batch.shape[0] * batch.shape[1]
                col = self.dim
                result = tensor.reshape(batch,
                                        newshape=[row, col],
                                        ndim=2)
                return space._cast(result, space.dtype)
            elif isinstance(space, space.Conv2DSpace):
                result = batch.dimshuffle(1, 0, 'x', 2)
                # Newly added part
                b01c_shape = [result.shape[0],
                              space.shape[0],
                              space.shape[1],
                              space.num_channels]
                result = result.flatten()
                result = tensor.reshape(result,
                                        newshape=b01c_shape,
                                        ndim=4)
                # end of newly added part

                return space._cast(result, space.dtype)
            else:
                print('Unexpected space', space)
                raise NotImplementedError

    @wraps(space.Space._check_sizes)
    def _check_sizes(self, space):
        """
        Called by self._format_as(space), to check whether self and space
        have compatible sizes. Throws a ValueError if they don't.
        """
        my_dimension = self.get_total_dimension()
        other_dimension = space.get_total_dimension()
        if my_dimension != other_dimension:
            if isinstance(space, Conv2DSpace):
                if my_dimension * space.shape[0] !=\
                        other_dimension:
                    raise ValueError(str(self)+" with total dimension " +
                                     str(my_dimension) +
                                     " can't format a batch into " +
                                     str(space) + "because its total dimension\
                                             is " +
                                     str(other_dimension))

    @wraps(space.Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)
        # Set a number which is not equal to batch_size for comfort debugging
        time_step = 5
        return np.zeros((time_step, batch_size, self.dim), dtype=dtype)

    @wraps(space.Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):

        dtype = self._clean_dtype_arg(dtype)
        broadcastable = [False] * 3
        broadcastable[1] = (batch_size == 1)
        broadcastable[2] = (self.dim == 1)
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable
                          )(name=name)

        if config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)

        return rval

    @wraps(space.Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # Data here is [time, batch, data]
        self.space._validate_impl(is_numeric, batch[0])

    @wraps(space.Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[1]

    @wraps(space.Space.get_total_dimension)
    def get_total_dimension(self):
        return self.space.get_total_dimension()


class SequenceMaskSpace(space.SimplyTypedSpace):
    """
    The mask is a binary matrix of size (time, batch) as floating
    point numbers, which can be multiplied with a hidden state to
    set all the elements which are out of the sequence to 0.

    Parameters
    ----------
        dtype : A numpy dtype string indicating this space's dtype.
    """
    def __init__(self, dtype='floatX'):
        self._dtype = self._clean_dtype_arg(dtype)
        super(SequenceMaskSpace, self).__init__(dtype)

    def __eq__(self, other):
        return isinstance(other, SequenceMaskSpace)

    def __hash__(self):
        return 1

    def __str__(self):
        return '%s' % (self.__class__.__name__)

    @wraps(space.Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[1]

    @wraps(space.Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if space == self:
            return batch
        else:
            raise NotImplementedError

    @wraps(space.Space.get_total_dimension)
    def get_total_dimension(self):
        return 0

    @wraps(space.Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        rval = tensor.matrix(name=name, dtype=dtype)
        if config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)
        return rval

    @wraps(space.Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        assert batch.ndim == 2

    @wraps(space.Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)
        # Set a number which is not equal to batch_size for comfort debugging
        time_step = 5
        # creating masks that have different zero values.
        # Non-zeros is 15 out of 20
        rval = np.zeros((time_step, batch_size), dtype=dtype)
        rval[:3, :1] = 1
        rval[:4, 1:] = 1
        return rval
