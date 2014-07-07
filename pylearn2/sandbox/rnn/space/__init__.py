import functools

from theano import tensor

from pylearn2 import space


class SequenceSpace(space.SimplyTypedSpace):
    """
    This space is simply a sequence of spaces over time. The first
    axis is generally time.

    Parameters
    ----------
    space : Space
        The space of which this is a sequence
    """
    def __init__(self, space):
        self.space = space
        self.dim = space.dim
        self._dtype = super(SequenceSpace, self)._clean_dtype_arg(space.dtype)

    @functools.wraps(space.Space.__eq__)
    def __eq__(self, other):
        if not isinstance(other, SequenceSpace):
            return False
        return self.space == other.space

    @functools.wraps(space.Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        return tensor.shape_padleft(
            self.space.make_theano_batch(name=name, dtype=dtype,
                                         batch_size=batch_size)
        )

    @functools.wraps(space.Space.get_total_dimension)
    def get_total_dimension(self):
        return self.space.get_total_dimension

    @functools.wraps(space.Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        self.space._validate_impl(is_numeric, batch[0])

    def __str__(self):
        return 'SequenceSpace(%s)' % (self.space)
