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

import functools
import numpy as np
import warnings

from theano import config
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
import theano.sparse
import theano.tensor as T
from theano.tensor import TensorType

from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX

if theano.sparse.enable_sparse:
    # We know scipy.sparse is available
    import scipy.sparse


class Space(object):
    """A vector space that can be transformed by a linear operator."""

    def __init__(self, validate_callbacks = None,
            np_validate_callbacks = None):
        """
        Initialize a Space.

        Parameters
        ----------
        validate_callbacks:
            Callbacks that are run at the start of a call to validate.
            Each should be a callable with the same signature as validate.
            An example use case is installing an instance-specific error
            handler that provides extra instructions for how to correct an
            input that is in a bad space.
        np_validate_callacks:
            similar to validate_callbacks, but run on calls to np_validate
        """

        if validate_callbacks is None:
            validate_callbacks = []

        if np_validate_callbacks is None:
            np_validate_callbacks = []

        self.__dict__.update(locals())
        del self.self

    def __ne__(self, other):
        """
        .. todo::

            WRITEME
        """
        return not (self == other)

    def __repr__(self):
        """
        .. todo::

            WRITEME
        """
        return str(self)

    def get_origin(self):
        """
        Returns the origin in this space.

        Returns
        -------
        origin : ndarray
            An NumPy array, the shape of a single points in this \
            space, representing the origin.
        """
        raise NotImplementedError()

    def get_origin_batch(self, n):
        """
        Returns a batch containing `n` copies of the origin.

        Returns
        -------
        batch : ndarray
            A NumPy array in the shape of a batch of `n` points in this \
            space (with points being indexed along the first axis), \
            each `batch[i]` being a copy of the origin.
        """
        raise NotImplementedError()

    def make_shared_batch(self, batch_size, name=None, dtype=None):
        """
        .. todo::

            WRITEME
        """
        if dtype is None:
            return sharedX(self.get_origin_batch(batch_size), name)
        else:
            raise NotImplementedError()

    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        """
        Returns a symbolic variable representing a batch of points
        in this space.

        Parameters
        ----------
        name : str
            WRITEME
        dtype : str
            WRITEME
        batch_size : int
            WRITEME

        Returns
        -------
        batch : TensorVariable
            A batch with the appropriate number of dimensions and \
            appropriate broadcast flags to represent a batch of \
            points in this space.
        """
        raise NotImplementedError()

    def make_batch_theano(self, name=None, dtype=None, batch_size=None):
        """ An alias to make_theano_batch """

        return self.make_theano_batch(name=name, dtype=dtype,
                                      batch_size=batch_size)

    def get_total_dimension(self):
        """
        Returns a Python int (not a theano iscalar) representing
        the dimensionality of a point in this space.

        If you format a batch of examples in this space as a
        design matrix (i.e., VectorSpace batch) then the
        number of columns will be equal to the total dimension.
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement get_total_dimension.")

    def np_format_as(self, batch, space):
        """
        Returns batch formatted to lie in space

        Should be invertible, i.e. batch should equal
        `space.format_as(self.format_as(batch, space), self)`

        Parameters
        ----------
        batch : numpy.ndarray
            Array which lies in the space represented by self
        space : Space
            WRITEME

        Returns
        -------
        WRITEME
        """
        raise NotImplementedError("%s does not implement np_format_as."
                                  % str(type(self)))

    def format_as(self, batch, space):
        """
        Returns batch formatted to lie in space

        Should be invertible, i.e. batch should equal
        `space.format_as(self.format_as(batch, space), self)`

        Parameters
        ----------
        batch : WRITEME
        space : Space
            WRITEME

        Returns
        -------
        WRITEME
        """

        self.validate(batch)

        my_dimension = self.get_total_dimension()
        other_dimension = space.get_total_dimension()

        if my_dimension != other_dimension:
            raise ValueError(str(self)+" with total dimension " +
                             str(my_dimension) +
                             " can't format a batch into " +
                             str(space) + "because its total dimension is " +
                             str(other_dimension))

        if self == space:
            rval = batch
        else:
            rval = self._format_as(batch, space)

        return rval

    def _format_as(self, batch, space):
        """
        Helper method that implements specifics of format_as for a particular
        subclass.

        Parameters
        ----------
        batch : numpy.ndarray
            Array which lies in the space represented by self
        space : Space
            WRITEME

        Returns
        -------
        WRITEME
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement _format_as.")

    def _meta_validate(self, batch, callbacks_name, validate):
        """
        Runs callbacks and validate on batch. Used to implement both
        validate and np_validate.

        Parameters
        ----------
        batch : any object, results in exception unless the object is a
                theano variable representing a batch of data in this
                Space.
        callbacks_name: name of field containing the callbacks
        validate: validate method to run
        """

        if not hasattr(self, callbacks_name):
            warnings.warn("It looks like the " + str(type(self)) +
                    "subclass of Space does not call the superclass __init__ "
                    "method. Currently this is a warning. It will become an "
                    "error on or after 2014-06-17.")
        else:
            callbacks = getattr(self, callbacks_name)
            for callback in callbacks:
                callback(batch)
        validate(batch)

    def validate(self, batch):
        """
        Runs all validate_callbacks.
        Raises an exception if batch is not a valid theano batch
        in this space.

        Parameters
        ----------
        batch : any object, results in exception unless the object is a
                theano variable representing a batch of data in this
                Space.
        """

        self._meta_validate(batch, 'validate_callbacks', self._validate)

    def _validate(self, batch):
        """
        Raises an exception if batch is not a valid theano batch
        in this space.

        Parameters
        ----------
        batch : any object, results in exception unless the object is a
                theano variable representing a batch of data in this
                Space.
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement _validate.")

    def np_validate(self, batch):
        """
        Runs all np_validate_callbacks.
        Raises an exception if batch is not a valid value for a batch in this
        space.

        Parameters
        ----------
        batch : any object, results in exception unless the object is a
                numpy representation of a batch of data in this
                Space.
        """

        self._meta_validate(batch, 'np_validate_callbacks', self._np_validate)

    def _np_validate(self, batch):
        """
        Raises an exception if batch is not a valid value for a batch in this
        space.

        Parameters
        ----------
        batch : WRITEME
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement _np_validate.")

    def batch_size(self, batch):
        """
        Read the batch size out of a symbolic batch.

        Parameters
        ----------
        batch : WRITEME
        """
        raise NotImplementedError(str(type(self)) + " does not implement " +
                                  "batch_size")

    def np_batch_size(self, batch):
        """
        Read the numeric batch size from a numeric (NumPy) batch.

        Parameters
        ----------
        batch : WRITEME
        """
        raise NotImplementedError(str(type(self)) + " does not implement " +
                                  "np_batch_size")

    def get_batch(self, data, start, end):
        """
        Returns a batch of data starting from index `start` to index `stop`

        Parameters
        ----------
        data : WRITEME
        start : WRITEME
        end : WRITEME
        """
        raise NotImplementedError(str(type(self)) + " does not implement " +
                                  "get_batch")


class VectorSpace(Space):
    """A space whose points are defined as fixed-length vectors."""
    def __init__(self, dim, sparse=False, **kwargs):
        """
        Initialize a VectorSpace.

        Parameters
        ----------
        dim : int
            Dimensionality of a vector in this space.
        sparse: bool
            Sparse vector or not
        kwargs: passed on to superclass constructor
        """

        super(VectorSpace, self).__init__(**kwargs)

        self.dim = dim
        self.sparse = sparse

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return '%(classname)s(dim=%(dim)s%(sparse)s)' % \
               dict(classname=self.__class__.__name__,
                    dim=self.dim,
                    sparse=(', sparse' if self.sparse else ''))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((self.dim,))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n, dtype=None):
        if dtype is None:
            dtype = config.floatX
        return np.zeros((n, self.dim), dtype=dtype)

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        self.validate(batch)
        return batch.shape[0]

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        self.np_validate(batch)
        return batch.shape[0]

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = config.floatX

        if self.sparse:
            if batch_size is not None:
                raise NotImplementedError("batch_size not implemented "
                                          "for sparse case")
            rval = theano.sparse.csr_matrix(name=name)
        else:
            if batch_size == 1:
                rval = T.row(name=name, dtype=dtype)
            else:
                rval = T.matrix(name=name, dtype=dtype)

        if config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(n=n)
        return rval

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.dim

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        self.np_validate(batch)
        return self._format_as(batch, space)

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):

        if isinstance(space, CompositeSpace):
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:, pos:pos+width]
                pos += width
                formatted = VectorSpace(width).format_as(subtensor, component)
                pieces.append(formatted)
            return tuple(pieces)

        if isinstance(space, Conv2DSpace):
            dims = {'b': batch.shape[0],
                    'c': space.num_channels,
                    0: space.shape[0],
                    1: space.shape[1]}
            if space.axes != space.default_axes:
                # Always use default_axes, so conversions like
                # Conv2DSpace(c01b) -> VectorSpace -> Conv2DSpace(b01c) work
                shape = [dims[ax] for ax in space.default_axes]
                batch = batch.reshape(shape)
                batch = batch.transpose(*[space.default_axes.index(ax)
                                          for ax in space.axes])
                return batch

            shape = tuple([dims[elem] for elem in space.axes])

            rval = batch.reshape(shape)

            return rval

        if isinstance(space, VectorSpace):
            if self.dim != space.dim:
                raise ValueError("Can't convert between VectorSpaces of "
                                 "different sizes (%d to %d)."
                                 % (self.dim, space.dim))
            if self.sparse != space.sparse:
                raise ValueError("Converting between sparse and non-sparse "
                                 "VectorSpaces not implemented.")
            return batch

        raise NotImplementedError(str(self) +
                                  " doesn't know how to format as " +
                                  str(space))

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return type(self) == type(other) and self.dim == other.dim

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), self.dim))

    def _validate(self, batch):
        """
        .. todo::

            WRITEME
        """
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("VectorSpace batch should be a theano Variable, "
                            "got " + str(type(batch)))
        if not self.sparse and not isinstance(batch.type,
                                              (theano.tensor.TensorType,
                                               CudaNdarrayType)):
            raise TypeError("VectorSpace batch should be TensorType or "
                            "CudaNdarrayType, got "+str(batch.type))
        if self.sparse and not isinstance(batch.type,
                                          theano.sparse.SparseType):
            raise TypeError()
        if batch.ndim != 2:
            raise ValueError('VectorSpace batches must be 2D, got %d '
                             'dimensions' % batch.ndim)
        for val in get_debug_values(batch):
            self.np_validate(val)

    @functools.wraps(Space._np_validate)
    def _np_validate(self, batch):
        # Use the 'CudaNdarray' string to avoid importing theano.sandbox.cuda
        # when it is not available
        if (not self.sparse
                and not isinstance(batch, np.ndarray)
                and type(batch) != 'CudaNdarray'):
            raise TypeError("The value of a VectorSpace batch should be a "
                            "numpy.ndarray, or CudaNdarray, but is %s."
                            % str(type(batch)))
        if self.sparse:
            if not theano.sparse.enable_sparse:
                raise TypeError("theano.sparse is not enabled, cannot have "
                                "a value for a sparse VectorSpace.")
            if not scipy.sparse.issparse(batch):
                raise TypeError("The value of a sparse VectorSpace batch "
                                "should be a sparse scipy matrix, got %s of "
                                "type %s." % (batch, type(batch)))
        if batch.ndim != 2:
            raise ValueError("The value of a VectorSpace batch must be "
                             "2D, got %d dimensions for %s." % (batch.ndim,
                                                                batch))
        if batch.shape[1] != self.dim:
            raise ValueError("The width of a VectorSpace batch must match "
                             "with the space's dimension, but batch has shape "
                             "%s and dim = %d." % (str(batch.shape), self.dim))


class Conv2DSpace(Space):
    """A space whose points are defined as (multi-channel) images."""

    # Assume pylearn2's get_topological_view format, since this is how
    # data is currently served up. If we make better iterators change
    # default to ('b', 'c', 0, 1) for theano conv2d
    default_axes = ('b', 0, 1, 'c')

    def __init__(self,
                 shape,
                 channels=None,
                 num_channels=None,
                 axes=None):
        """
        Initialize a Conv2DSpace.

        Parameters
        ----------
        shape : sequence, length 2
            The shape of a single image, i.e. (rows, cols).
        num_channels: int     (synonym: channels)
            Number of channels in the image, i.e. 3 if RGB.
        axes: A tuple indicating the semantics of each axis.
                'b' : this axis is the batch index of a minibatch.
                'c' : this axis the channel index of a minibatch.
                <i> : this is topological axis i (i.e., 0 for rows, 1 for \
                cols) \
                \
                For example, a PIL image has axes (0, 1, 'c') or (0, 1). \
                The pylearn2 image displaying functionality uses \
                    ('b', 0, 1, 'c') for batches and (0, 1, 'c') for images. \
                theano's conv2d operator uses ('b', 'c', 0, 1) images.
        """

        assert (channels is None) + (num_channels is None) == 1
        if num_channels is None:
            num_channels = channels

        assert isinstance(num_channels, py_integer_types)

        if not hasattr(shape, '__len__') or len(shape) != 2:
            raise ValueError("shape argument to Conv2DSpace must be length 2")
        assert all(isinstance(elem, py_integer_types) for elem in shape)
        assert all(elem > 0 for elem in shape)
        assert isinstance(num_channels, py_integer_types)
        assert num_channels > 0
        # Convert shape to a tuple, so it can be hashable, and self can be too
        self.shape = tuple(shape)
        self.num_channels = num_channels
        if axes is None:
            axes = self.default_axes
        assert len(axes) == 4
        self.axes = tuple(axes)

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return "Conv2DSpace{shape=%s,num_channels=%d}" % (str(self.shape),
                                                          self.num_channels)

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and
                self.shape == other.shape and
                self.num_channels == other.num_channels
                and tuple(self.axes) == tuple(other.axes))

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), self.shape, self.num_channels, self.axes))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = {0: self.shape[0], 1: self.shape[1], 'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes if elem != 'b']
        return np.zeros(shape)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n, dtype=None):
        if dtype is None:
            dtype = config.floatX

        if not isinstance(n, py_integer_types):
            raise TypeError("Conv2DSpace.get_origin_batch expects an int, "
                            "got " + str(n) + " of type " + str(type(n)))
        assert n > 0
        dims = {'b': n,
                0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes]
        return np.zeros(shape, dtype=dtype)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = config.floatX

        broadcastable = [False] * 4
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = (batch_size == 1)
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
            rval.tag.test_value = self.get_origin_batch(n=n)
        return rval

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        self.validate(batch)
        return batch.shape[self.axes.index('b')]

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        self.np_validate(batch)
        return batch.shape[self.axes.index('b')]

    @staticmethod
    def convert(tensor, src_axes, dst_axes):
        """
        Returns a view of tensor using the axis semantics defined
        by dst_axes. (If src_axes matches dst_axes, returns
        tensor itself)

        Useful for transferring tensors between different
        Conv2DSpaces.

        Parameters
        ----------
        tensor : tensor_like
            A 4-tensor representing a batch of images
        src_axes : WRITEME
            Axis semantics of tensor
        dst_axes : WRITEME
            WRITEME
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 4
        assert len(dst_axes) == 4

        if src_axes == dst_axes:
            return tensor

        shuffle = [src_axes.index(elem) for elem in dst_axes]

        return tensor.dimshuffle(*shuffle)

    @staticmethod
    def convert_numpy(tensor, src_axes, dst_axes):
        """
        Returns a view of tensor using the axis semantics defined
        by dst_axes. (If src_axes matches dst_axes, returns
        tensor itself)

        Useful for transferring tensors between different
        Conv2DSpaces.

        Parameters
        ----------
        tensor : numpy.ndarray
            A 4-tensor representing a batch of images
        src_axes : WRITEME
            Axis semantics of tensor
        dst_axes : WRITEME
            WRITEME
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 4
        assert len(dst_axes) == 4

        if src_axes == dst_axes:
            return tensor

        shuffle = [src_axes.index(elem) for elem in dst_axes]

        return tensor.transpose(*shuffle)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):

        # Patch old pickle files
        if not hasattr(self, 'num_channels'):
            self.num_channels = self.nchannels

        return self.shape[0] * self.shape[1] * self.num_channels

    @functools.wraps(Space._validate)
    def _validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("Conv2DSpace batches must be theano Variables, "
                            "got "+str(type(batch)))
        if not isinstance(batch.type, (theano.tensor.TensorType,
                                       CudaNdarrayType)):
            raise TypeError()
        if batch.ndim != 4:
            raise ValueError()
        for val in get_debug_values(batch):
            self.np_validate(val)

    @functools.wraps(Space._np_validate)
    def _np_validate(self, batch):
        if (not isinstance(batch, np.ndarray)
                and type(batch) != 'CudaNdarray'):
            raise TypeError("The value of a Conv2DSpace batch should be a "
                            "numpy.ndarray, or CudaNdarray, but is %s."
                            % str(type(batch)))
        if batch.ndim != 4:
            raise ValueError("The value of a Conv2DSpace batch must be "
                             "4D, got %d dimensions for %s." %
                             (batch.ndim, batch))

        d = self.axes.index('c')
        actual_channels = batch.shape[d]
        if actual_channels != self.num_channels:
            raise ValueError("Expected axis %d to be number of channels (%d) "
                             "but it is %d" %
                             (d, self.num_channels, actual_channels))
        assert batch.shape[self.axes.index('c')] == self.num_channels

        for coord in [0, 1]:
            d = self.axes.index(coord)
            actual_shape = batch.shape[d]
            expected_shape = self.shape[coord]
            if actual_shape != expected_shape:
                raise ValueError("Conv2DSpace with shape %s and axes %s "
                                 "expected dimension %s of a batch (%s) to "
                                 "have length %s but it has %s"
                                 % (str(self.shape), str(self.axes), str(d),
                                    str(batch), str(expected_shape),
                                    str(actual_shape)))

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        self.np_validate(batch)
        if isinstance(space, VectorSpace):
            # We need to ensure that the resulting batch will always be
            # the same in `space`, no matter what the axes of `self` are.
            if self.axes != self.default_axes:
                # The batch index goes on the first axis
                assert self.default_axes[0] == 'b'
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in self.default_axes])
            return batch.reshape((batch.shape[0], self.get_total_dimension()))
        if isinstance(space, Conv2DSpace):
            return Conv2DSpace.convert_numpy(batch, self.axes, space.axes)
        raise NotImplementedError("%s doesn't know how to format as %s"
                                  % (str(self), str(space)))

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        self.validate(batch)
        if isinstance(space, VectorSpace):
            # We need to ensure that the resulting batch will always be
            # the same in `space`, no matter what the axes of `self` are.
            if self.axes != self.default_axes:
                # The batch index goes on the first axis
                assert self.default_axes[0] == 'b'
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in self.default_axes])
            return batch.reshape((batch.shape[0], self.get_total_dimension()))
        if isinstance(space, Conv2DSpace):
            return Conv2DSpace.convert(batch, self.axes, space.axes)
        raise NotImplementedError("%s doesn't know how to format as %s"
                                  % (str(self), str(space)))


class CompositeSpace(Space):
    """A Space whose points are tuples of points in other spaces """
    def __init__(self, components):
        """
        .. todo::

            WRITEME
        """
        assert isinstance(components, (list, tuple))
        self.num_components = len(components)
        for i, component in enumerate(components):
            if not isinstance(component, Space):
                raise TypeError("component %d is %s of type %s, expected "
                                "Space instance. " %
                                (i, str(component), str(type(component))))
        self.components = list(components)

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and
                len(self.components) == len(other.components) and
                all([my_component == other_component for
                     my_component, other_component in
                     zip(self.components, other.components)]))

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), tuple(self.components)))

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return '%(classname)s(%(components)s)' % \
               dict(classname=self.__class__.__name__,
                    components=', '.join([str(c) for c in self.components]))

    def restrict(self, subset):
        """
        Returns a new Space containing only the components whose indices
        are given in subset.

        The new space will contain the components in the order given in the
        subset list.

        Parameters
        ----------
        subset : WRITEME

        Notes
        -----
        The returned Space may not be a CompositeSpace if `subset` contains
        only one index.
        """

        assert isinstance(subset, (list, tuple))

        if len(subset) == 1:
            idx, = subset
            return self.components[idx]

        return CompositeSpace([self.components[i] for i in subset])

    def restrict_batch(self, batch, subset):
        """
        Returns a batch containing only the components whose indices are
        present in subset. May not be a tuple anymore if there is only one
        index. Outputs will be ordered in the order that they appear in subset.

        Parameters
        ----------
        batch : WRITEME
        subset : WRITEME
        """

        self.validate(batch)
        assert isinstance(subset, (list, tuple))

        if len(subset) == 1:
            idx, = subset
            return batch[idx]

        return tuple([batch[i] for i in subset])

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return sum([component.get_total_dimension() for component in
                    self.components])

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        """
        Supports formatting to a single VectorSpace, or to a CompositeSpace.

        CompositeSpace->VectorSpace:
          Traverses the nested components in depth-first order, serializing the
          leaf nodes (i.e. the non-composite subspaces) into the VectorSpace.

        CompositeSpace->CompositeSpace:

          Only works for two CompositeSpaces that have the same nested
          structure. Traverses both CompositeSpaces' nested components in
          parallel, converting between corresponding non-composite components
          in <self> and <space> as:

              `self_component.np_format_as(batch_component, space_component)`

        Parameters
        ----------
        batch : WRITEME
        space : WRITEME

        Returns
        -------
        WRITEME
        """
        self.np_validate(batch)
        if isinstance(space, VectorSpace):
            pieces = []
            for component, input_piece in zip(self.components, batch):
                width = component.get_total_dimension()
                pieces.append(component.np_format_as(input_piece,
                                                     VectorSpace(width)))
            return np.concatenate(pieces, axis=1)

        if isinstance(space, CompositeSpace):
            def recursive_np_format_as(orig_space, batch, dest_space):
                if not (isinstance(orig_space, CompositeSpace) ==
                        isinstance(dest_space, CompositeSpace)):
                    raise TypeError("Can't convert between CompositeSpaces "
                                    "with different tree structures")

                # No need to check batch's tree structure; np_validate has
                # already done that above.

                if isinstance(orig_space, CompositeSpace):
                    return tuple(recursive_np_format_as(os, bt, ds)
                                 for os, bt, ds
                                 in safe_zip(orig_space.components,
                                             batch,
                                             dest_space.components))
                else:
                    return orig_space.np_format_as(batch, dest_space)

            return recursive_np_format_as(self, batch, space)

        raise NotImplementedError(str(self) +
                                  " does not know how to format as " +
                                  str(space))

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        """
        Supports formatting to a single VectorSpace, or to a CompositeSpace.

        CompositeSpace->VectorSpace:
          Traverses the nested components in depth-first order, serializing the
          leaf nodes (i.e. the non-composite subspaces) into the VectorSpace.

        CompositeSpace->CompositeSpace:

          Only works for two CompositeSpaces that have the same nested
          structure. Traverses both CompositeSpaces' nested components in
          parallel, converting between corresponding non-composite components
          in <self> and <space> as:

              `self_component.format_as(batch_component, space_component)`

        Parameters
        ----------
        batch : WRITEME
        space : WRITEME

        Returns
        -------
        WRITEME
        """
        if isinstance(space, VectorSpace):
            pieces = []
            for component, input_piece in zip(self.components, batch):
                width = component.get_total_dimension()
                pieces.append(component.format_as(input_piece,
                                                  VectorSpace(width)))
            return T.concatenate(pieces, axis=1)

        if isinstance(space, CompositeSpace):
            def recursive_format_as(orig_space, batch, dest_space):
                if not (isinstance(orig_space, CompositeSpace) ==
                        isinstance(dest_space, CompositeSpace)):
                    raise TypeError("Can't convert between CompositeSpaces "
                                    "with different tree structures")

                # No need to check batch's tree structure; validate() has
                # already done that, in Space.format_as(), which called
                # CompositeSpace._format_as()

                if isinstance(orig_space, CompositeSpace):
                    return tuple(recursive_format_as(os, bt, ds)
                                 for os, bt, ds
                                 in safe_zip(orig_space.components,
                                             batch,
                                             dest_space.components))
                else:
                    return orig_space.format_as(batch, dest_space)

            return recursive_format_as(self, batch, space)

        raise NotImplementedError(str(self) +
                                  " does not know how to format as " +
                                  str(space))

    @functools.wraps(Space._validate)
    def _validate(self, batch):
        if not isinstance(batch, tuple):
            raise TypeError()
        if len(batch) != self.num_components:
            raise ValueError("Expected "+str(self.num_components) +
                             " elements in batch, got " + str(len(batch)))
        for batch_elem, component in zip(batch, self.components):
            component.validate(batch_elem)

    @functools.wraps(Space._np_validate)
    def _np_validate(self, batch):
        if not isinstance(batch, tuple):
            raise TypeError("The value of a CompositeSpace batch should be a "
                            "tuple, but is %s of type %s." %
                            (batch, type(batch)))
        if len(batch) != self.num_components:
            raise ValueError("Expected %d elements in batch, got %d"
                             % (self.num_components, len(batch)))
        for batch_elem, component in zip(batch, self.components):
            component.np_validate(batch_elem)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        return tuple([component.get_origin_batch(n) for
                      component in self.components])

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if name is None:
            name = [None] * len(self.components)
        elif not isinstance(name, (list, tuple)):
            name = ['%s[%i]' % (name, i) for i in xrange(len(self.components))]

        if dtype is None:
            dtype = [None] * len(self.components)
        assert isinstance(name, (list, tuple))
        assert isinstance(dtype, (list, tuple))

        rval = tuple([x.make_theano_batch(name=n,
                                          dtype=d,
                                          batch_size=batch_size)
                      for x, n, d in safe_zip(self.components,
                                              name,
                                              dtype)])
        return rval

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        # All components should have the same effective batch size,
        # with the exeption of NullSpace, and CompositeSpace with
        # 0 components, which will return 0, because they do not
        # represent any data.
        self.validate(batch)

        for c, d in safe_zip(self.components, batch):
            b = c.batch_size(d)

            if b != 0:
                # We assume they are all equal to b
                return b

        # All components are empty
        return 0

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        self.np_validate(batch)
        # We actually check that all non-zero batch sizes are equal
        rval = 0
        for c, d in safe_zip(self.components, batch):
            b = c.np_batch_size(d)
            if b != 0:
                if rval == 0:
                    # First non-zero value we encounter, this is our candidate
                    rval = b
                elif b != rval:
                    raise ValueError("All non-empty components of a "
                                     "CompositeSpace should have the same "
                                     "batch size, but we encountered "
                                     "components with size %d, then %d." %
                                     (rval, b))
        return rval


class NullSpace(Space):
    """
    A space that contains no data.

    When symbolic or numerical data for that space actually has to be
    represented, None is used as a placeholder.

    The source associated to that Space is the empty string ('').
    """

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return "NullSpace"

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return type(self) == type(other)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash(type(self))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None):
        return None

    @functools.wraps(Space._validate)
    def _validate(self, batch):
        if batch is not None:
            raise TypeError("NullSpace only accepts 'None' as a "
                            "place-holder for data, not %s of type %s"
                            % (batch, type(batch)))

    @functools.wraps(Space._np_validate)
    def _np_validate(self, batch):
        if batch is not None:
            raise TypeError("NullSpace only accepts 'None' as a "
                            "place-holder for data, not %s of type %s"
                            % (batch, type(batch)))

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        self.np_validate(batch)
        assert isinstance(space, NullSpace)
        return None

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        self.validate(batch)
        assert isinstance(space, NullSpace)
        return None

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        # There is no way to know how many examples would actually
        # have been in the batch, since it is empty. We return 0.
        self.validate(batch)
        return 0

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        # There is no way to know how many examples would actually
        # have been in the batch, since it is empty. We return 0.
        self.np_validate(batch)
        return 0
