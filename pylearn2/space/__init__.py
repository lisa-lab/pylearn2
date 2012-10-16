"""

Classes that define how vector spaces are formatted

Most of our models can be viewed as linearly transforming
one vector space to another. These classes define how the
vector spaces should be represented as theano/numpy
variables.

For example, the VectorSpace class just represents a
vector space with a vector, and the model can transform
between spaces with a matrix multiply. The Conv2DSpace
represents a vector space as an image, and the model
can transform between spaces with a 2D convolution.

To make models as general as possible, models should be
written in terms of Spaces, rather than in terms of
numbers of hidden units, etc. The model should also be
written to transform between spaces using a generic
linear transformer from the pylearn2.linear module.

The Space class is needed so that the model can specify
what kinds of inputs it needs and what kinds of outputs
it will produce when communicating with other parts of
the library. The model also uses Space objects internally
to allocate parameters like hidden unit bias terms in
the right space.

"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import theano.tensor as T
import theano.sparse
from theano.tensor import TensorType
from theano import config
import functools
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType


class Space(object):
    """A vector space that can be transformed by a linear operator."""
    def get_origin(self):
        """
        Returns the origin in this space.

        Returns
        -------
        origin : ndarray
            An NumPy array, the shape of a single points in this
            space, representing the origin.
        """
        raise NotImplementedError()

    def get_origin_batch(self, n):
        """
        Returns a batch containing `n` copies of the origin.

        Returns
        -------
        batch : ndarray
            A NumPy array in the shape of a batch of `n` points in this
            space (with points being indexed along the first axis),
            each `batch[i]` being a copy of the origin.
        """
        raise NotImplementedError()

    def make_theano_batch(self, name=None, dtype=None):
        """
        Returns a symbolic variable representing a batch of points
        in this space.

        Returns
        -------
        batch : TensorVariable
            A batch with the appropriate number of dimensions and
            appropriate broadcast flags to represent a batch of
            points in this space.
        """
        raise NotImplementedError()

    def make_batch_theano(self, name = None, dtype = None):
        """ An alias to make_theano_batch """

        return self.make_theano_batch(name = name, dtype = dtype)

    def get_total_dimension(self):
        """
        Returns a Python int (not a theano iscalar) representing
        the dimensionality of a point in this space.

        If you format a batch of examples in this space as a
        design matrix (i.e., VectorSpace batch) then the
        number of columns will be equal to the total dimension.
        """

        raise NotImplementedError(str(type(self))+" does not implement get_total_dimension.")

    def format_as(self, batch, space):
        """
        batch: a theano batch which lies in the space represented by self
        space: a Space

        returns batch formatted to lie in space

        Should be invertible, i.e.
        batch should equal
        space.format_as(self.format_as(batch, space), self)
        """

        assert self.get_total_dimension() == space.get_total_dimension()

        if self == space:
            rval = batch
        else:
            rval = self._format_as(batch, space)

        return rval

    def _format_as(self, batch, space):
        """
        Helper method that implements specifics of format_as for a particular subclass.
        """

        raise NotImplementedError(str(type(self))+" does not implement _format_as.")

    def validate(self, batch):
        """ Raises an exception if batch is not a valid theano batch
        in this space. """

        raise NotImplementedError(str(type(self))+" does not implement validate.")

class VectorSpace(Space):
    """A space whose points are defined as fixed-length vectors."""
    def __init__(self, dim, sparse=False):
        """
        Initialize a VectorSpace.

        Parameters
        ----------
        dim : int
            Dimensionality of a vector in this space.
        sparse: bool
            Sparse vector or not
        """
        self.dim = dim
        self.sparse = sparse

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((self.dim,))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        return np.zeros((n, self.dim))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None):
        if dtype is None:
            dtype = config.floatX

        if self.sparse:
            return theano.sparse.csr_matrix(name=name)
        else:
            return T.matrix(name=name, dtype=dtype)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.dim

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):

        if isinstance(space, CompositeSpace):
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:,pos:pos+width]
                pos += width
                formatted = VectorSpace(width).format_as(subtensor, component)
                pieces.append(formatted)
            return tuple(pieces)

        if isinstance(space, Conv2DSpace):
            if space.axes[0] != 'b':
                raise NotImplementedError("Will need to reshape to ('b',*) then do a dimshuffle. Be sure to make this the inverse of space._format_as(x, self)")
            dims = { 'b' : batch.shape[0], 'c' : space.nchannels, 0 : space.shape[0], 1 : space.shape[1] }

            shape = tuple( [ dims[elem] for elem in space.axes ] )

            rval = batch.reshape(shape)

            return rval

        raise NotImplementedError("VectorSpace doesn't know how to format as "+str(type(space)))

    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("VectorSpace batch should be a theano Variable, got "+str(type(batch)))
        if not self.sparse and not isinstance(batch.type, (theano.tensor.TensorType, CudaNdarrayType)):
            raise TypeError("VectorSpace batch should be TensorType or CudaNdarrayType, got "+str(batch.type))
        if self.sparse and not isinstance(batch.type, theano.sparse.SparseType):
            raise TypeError()
        if batch.ndim != 2:
            raise ValueError()

class Conv2DSpace(Space):
    """A space whose points are defined as (multi-channel) images."""
    def __init__(self, shape, nchannels, axes = None):
        """
        Initialize a Conv2DSpace.

        Parameters
        ----------
        shape : sequence, length 2
            The shape of a single image, i.e. (rows, cols).
        nchannels: int
            Number of channels in the image, i.e. 3 if RGB.
        axes: A tuple indicating the semantics of each axis.
                'b' : this axis is the batch index of a minibatch.
                'c' : this axis the channel index of a minibatch.
                <i>  : this is topological axis i (i.e., 0 for rows,
                                  1 for cols)

                For example, a PIL image has axes (0, 1, 'c') or (0, 1).
                The pylearn2 image displaying functionality uses
                    ('b', 0, 1, 'c') for batches and (0, 1, 'c') for images.
                theano's conv2d operator uses ('b', 'c', 0, 1) images.
        """
        if not hasattr(shape, '__len__') or len(shape) != 2:
            raise ValueError("shape argument to Conv2DSpace must be length 2")
        self.shape = shape
        self.nchannels = nchannels
        if axes is None:
            # Assume pylearn2's get_topological_view format, since this is how
            # data is currently served up. If we make better iterators change
            # default to ('b', 'c', 0, 1) for theano conv2d
            axes = ('b', 0, 1, 'c')
        assert len(axes) == 4
        self.axes = axes

    def __eq__(self, other):
        return type(self) == type(other) and \
                self.shape == other.shape and \
                self.nchannels == other.nchannels \
                and self.axes == other.axes

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = { 0: self.shape[0], 1: self.shape[1], 'c' : self.nchannels }
        shape = [ dims[elem] for elem in self.axes if elem != 'b' ]
        return np.zeros(shape)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        dims = { 'b' : n, 0: self.shape[0], 1: self.shape[1], 'c' : self.nchannels }
        shape = [ dims[elem] for elem in self.axes ]
        return np.zeros(shape)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None):
        if dtype is None:
            dtype = config.floatX
        return TensorType(dtype=dtype,
                          broadcastable=(False, False, False,
                                         self.nchannels == 1)
                         )(name=name)

    @staticmethod
    def convert(tensor, src_axes, dst_axes):
        """
            tensor: a 4 tensor representing a batch of images

            src_axes: the axis semantics of tensor

            Returns a view of tensor using the axis semantics defined
            by dst_axes. (If src_axes matches dst_axes, returns
            tensor itself)

            Useful for transferring tensors between different
            Conv2DSpaces.
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 4
        assert len(dst_axes) == 4

        if src_axes == dst_axes:
            return tensor

        shuffle = [ src_axes.index(elem) for elem in dst_axes ]

        return tensor.dimshuffle(*shuffle)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.shape[0] * self.shape[1] * self.nchannels

    @functools.wraps(Space.validate)
    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError()
        if not isinstance(batch.type, (theano.tensor.TensorType,CudaNdarrayType)):
            raise TypeError()
        if batch.ndim != 4:
            raise ValueError()
        for val in get_debug_values(batch):
            assert val.shape[self.axes.index('c')] == self.nchannels
            for coord in [0,1]:
                assert val.shape[self.axes.index(coord)] == self.shape[coord]

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        self.validate(batch)
        if isinstance(space, VectorSpace):
            if self.axes[0] != 'b':
                raise NotImplementedError("Need to dimshuffle so b is first axis before reshape")
            return batch.reshape((batch.shape[0], self.get_total_dimension()))
        if isinstance(space, Conv2DSpace):
            return Conv2DSpace.convert(batch, self.axes, space.axes)
        raise NotImplementedError("Conv2DSPace doesn't know how to format as "+str(type(space)))

class CompositeSpace(Space):
    """A Space whose points are tuples of points in other spaces """
    def __init__(self, components):
        assert isinstance(components, (list, tuple))
        self.num_components = len(components)
        assert all([isinstance(component, Space) for component in components])
        self.components = list(components)

    def __eq__(self, other):
        return type(self) == type(other) and \
            len(self.components) == len(other.components) and \
            all([my_component == other_component for
                my_component, other_component in \
                zip(self.my_components, other.components)])

    def restrict(self, subset):
        """Returns a new Space containing only the components whose indices
        are given in subset.

        The new space will contain the components in the order given in the
        subset list.

        Note that the returned Space may not be a CompositeSpace if subset
        contains only one index.
        """

        assert isinstance(subset, (list, tuple))

        if len(subset) == 1:
            idx, = subset
            return self.components[idx]

        return CompositeSpace([self.components[i] for i in subset])

    def restrict_batch(self, batch, subset):
        """Returns a batch containing only the components whose indices are present
        in subset. May not be a tuple anymore if there is only one index. Outputs
        will be ordered in the order that they appear in subset."""

        self.validate(batch)
        assert isinstance(subset, (list, tuple))

        if len(subset) == 1:
            idx, = subset
            return batch[idx]

        return tuple([batch[idx] for idx in subset])

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return sum([component.get_total_dimension() for component in
            self.components])

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        if isinstance(space, VectorSpace):
            pieces = []
            for component, input_piece in zip(self.components, batch):
                width = component.get_total_dimension()
                pieces.append(component.format_as(input_piece, VectorSpace(width)))
            return T.concatenate(pieces, axis=1)

        raise NotImplementedError("CompositeSpace does not know how to format as "+str(space))

    @functools.wraps(Space.validate)
    def validate(self, batch):
        if not isinstance(batch, tuple):
            raise TypeError()
        if len(batch) != self.num_components:
            raise ValueError()
        for batch_elem, component in zip(batch, self.components):
            component.validate(batch_elem)

