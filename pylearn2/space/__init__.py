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

import numpy as np
import theano.tensor as T
import theano.sparse
from theano.tensor import TensorType
from theano import config
import functools


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

