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
import ipdb
import functools, warnings
import numpy as np
import theano
import theano.sparse
from theano import tensor
from theano.tensor import TensorType
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
from pylearn2.utils import py_integer_types, safe_zip, sharedX, wraps
from pylearn2.format.target_format import OneHotFormatter

if theano.sparse.enable_sparse:
    # We know scipy.sparse is available
    import scipy.sparse


def _is_batch_all(batch, predicate):
    """
    Implementation of is_symbolic_batch() and is_numeric_batch().
    Returns True iff predicate() returns True for all components of
    (possibly composite) batch.

    Parameters
    ----------
    batch : any numeric or symbolic batch.
        This includes numpy.ndarray, theano.gof.Variable, None, or a (nested)
        tuple thereof.

    predicate : function.
        A unary function of any non-composite batch that returns True or False.
    """
    # Catches any CompositeSpace batches that were mistakenly hand-constructed
    # using nested lists rather than nested tuples.
    assert not isinstance(batch, list)

    # Data-less batches such as None or () are valid numeric and symbolic
    # batches.
    #
    # Justification: we'd like
    # is_symbolic_batch(space.make_theano_batch()) to always be True, even if
    # space is an empty CompositeSpace.
    if batch is None or (isinstance(batch, tuple) and len(batch) == 0):
        return True

    if isinstance(batch, tuple):
        subbatch_results = tuple(_is_batch_all(b, predicate)
                                 for b in batch)
        result = all(subbatch_results)

        # The subbatch_results must be all true, or all false, not a mix.
        assert result == any(subbatch_results), ("composite batch had a "
                                                 "mixture of numeric and "
                                                 "symbolic subbatches. This "
                                                 "should never happen.")
        return result
    else:
        return predicate(batch)


def is_symbolic_batch(batch):
    """
    Returns True if batch is a symbolic variable.

    Note that a batch may be both a symbolic and numeric variable
    (e.g. () for empty CompositeSpaces, None for NullSpaces).
    """

    return _is_batch_all(batch, lambda x: isinstance(x, theano.gof.Variable))


def is_numeric_batch(batch):
    """
    Returns True if batch is a numeric variable.

    Note that a batch may be both a symbolic and numeric variable
    (e.g. () for empty CompositeSpaces, None for NullSpaces).
    """
    def is_numeric(batch):
        return isinstance(batch, np.ndarray) or scipy.sparse.issparse(batch)

    return _is_batch_all(batch, is_numeric)


def _dense_to_sparse(batch):
    """
    Casts dense batches to sparse batches (non-composite).

    Supports both symbolic and numeric variables.
    """
    if isinstance(batch, tuple):
        raise TypeError("Composite batches not supported.")

    assert not isinstance(batch, list)

    if is_symbolic_batch(batch):
        assert isinstance(batch, theano.tensor.TensorVariable)
        return theano.sparse.csr_from_dense(batch)
    else:
        assert isinstance(batch, np.ndarray), "type of batch: %s" % type(batch)
        return scipy.sparse.csr_matrix(batch)


def _reshape(arg, shape):
    """
    Reshapes a tensor. Supports both symbolic and numeric variables.

    This is a hack that first converts from sparse to dense, reshapes
    the dense tensor, then re-converts from dense to sparse. It is
    therefore memory-inefficient and unsuitable for large tensors. It
    will be replaced by a proper sparse reshaping Op once Theano
    implements that.
    """

    if isinstance(arg, tuple):
        raise TypeError("Composite batches not supported.")

    assert not isinstance(arg, list)

    if isinstance(arg, (np.ndarray, theano.tensor.TensorVariable)):
        return arg.reshape(shape)
    elif isinstance(arg, theano.sparse.SparseVariable):
        warnings.warn("Using pylearn2.space._reshape(), which is a "
                      "memory-inefficient hack for reshaping sparse tensors. "
                      "Do not use this on large tensors. This will eventually "
                      "be replaced by a proper Theano Op for sparse "
                      "reshaping, once that is written.")
        dense = theano.sparse.dense_from_sparse(arg)
        dense = dense.reshape(shape)
        if arg.format == 'csr':
            return theano.sparse.csr_from_dense(dense)
        elif arg.format == 'csc':
            return theano.sparse.csc_from_dense(dense)
        else:
            raise ValueError('Unexpected sparse format "%s".' % arg.format)
    else:
        raise TypeError('Unexpected batch type "%s"' % str(type(arg)))


def _cast(arg, dtype):
    """
    Does element-wise casting to dtype.
    Supports symbolic, numeric, simple, and composite batches.

    Returns <arg> untouched if <dtype> is None, or dtype is unchanged
    (i.e. casting a float32 batch to float32).

       (One exception: composite batches are never returned as-is.
        A new tuple will always be returned. However, any components
        with unchanged dtypes will be returned untouched.)
    """

    if dtype is None:
        return arg

    assert dtype in tuple(t.dtype for t in theano.scalar.all_types)

    if isinstance(arg, tuple):
        return tuple(_cast(a, dtype) for a in arg)
    elif isinstance(arg, np.ndarray):
        # theano._asarray is a safer drop-in replacement to numpy.asarray.
        return theano._asarray(arg, dtype=dtype)
    elif str(type(arg)) == "<type 'CudaNdarray'>":  # numeric CUDA array
        if str(dtype) != 'float32':
            raise TypeError("Can only cast a numeric CudaNdarray to "
                            "float32, not %s" % dtype)
        else:
            return arg
    elif (isinstance(arg, theano.gof.Variable) and
          isinstance(arg.type, CudaNdarrayType)):  # symbolic CUDA array
        if str(dtype) != 'float32':
            raise TypeError("Can only cast a theano CudaNdArrayType to "
                            "float32, not %s" % dtype)
        else:
            return arg
    elif scipy.sparse.issparse(arg):
        return arg.astype(dtype)
    elif isinstance(arg, theano.tensor.TensorVariable):
        return theano.tensor.cast(arg, dtype)
    elif isinstance(arg, theano.sparse.SparseVariable):
        return theano.sparse.cast(arg, dtype)
    elif isinstance(arg, theano.sandbox.cuda.var.CudaNdarrayVariable):
        return arg
    else:
        raise TypeError("Unsupported arg type '%s'" % str(type(arg)))


class Space(object):
    """
    A vector space that can be transformed by a linear operator.

    Space and its subclasses are used to transform a data batch's geometry
    (e.g. vectors <--> matrices) and optionally, its dtype (e.g. float <-->
    int).

    Batches may be one of the following types:

        - numpy.ndarray
        - scipy.sparse.csr_matrix
        - theano.gof.Variable
        - None (for NullSpace)
        - A (nested) tuple of the above, possibly empty
          (for CompositeSpace).

    Parameters
    ----------
    validate_callbacks : list
        Callbacks that are run at the start of a call to validate.
        Each should be a callable with the same signature as validate.
        An example use case is installing an instance-specific error
        handler that provides extra instructions for how to correct an
        input that is in a bad space.
    np_validate_callacks : list
        similar to validate_callbacks, but run on calls to np_validate
    """
    def __init__(self, validate_callbacks=None,
                 np_validate_callbacks=None):
        if validate_callbacks is None:
            validate_callbacks = []

        if np_validate_callbacks is None:
            np_validate_callbacks = []

        self.validate_callbacks = validate_callbacks
        self.np_validate_callbacks = np_validate_callbacks

    # Forces subclasses to implement __eq__.
    # This is necessary for _format_as to work correctly.
    def __eq__(self, other):
        """
        Returns true iff
        space.format_as(batch, self) and
        space.format_as(batch, other) return the same formatted batch.
        """
        raise NotImplementedError("__eq__ not implemented in class %s." %
                                  type(self))

    def get_batch_axis(self):
        """
        Returns the batch axis of the output space.

        Return
        ------
        batch_axis : int
            the axis of the batch in the output space.
        """
        return 0

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

    @property
    def dtype(self):
        """
        An object representing the data type used by this space.

        For simple spaces, this will be a dtype string, as used by numpy,
        scipy, and theano (e.g. 'float32').

        For data-less spaces like NoneType, this will be some other string.

        For composite spaces, this will be a nested tuple of such strings.
        """
        raise NotImplementedError()

    @dtype.setter
    def dtype(self, new_value):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    @dtype.deleter
    def dtype(self):
        """
        .. todo::

            WRITEME
        """
        raise RuntimeError("You may not delete the dtype of a space, "
                           "though you can set it to None.")

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

    def get_origin_batch(self, batch_size, dtype=None):
        """
        Returns a batch containing `batch_size` copies of the origin.

        Parameters
        ----------
        batch_size : int
            The number of examples in the batch to be returned.
        dtype : WRITEME
            The dtype of the batch to be returned. Default = None.
            If None, use self.dtype.

        Returns
        -------
        batch : ndarray
            A NumPy array in the shape of a batch of `batch_size` points in
            this space (with points being indexed along the first axis),
            each `batch[i]` being a copy of the origin.
        """
        raise NotImplementedError()

    def make_shared_batch(self, batch_size, name=None, dtype=None):
        """
        .. todo::

            WRITEME
        """

        dtype = self._clean_dtype_arg(dtype)
        origin_batch = self.get_origin_batch(batch_size, dtype)
        return theano.shared(origin_batch, name=name)

    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        """
        Returns a symbolic variable representing a batch of points
        in this space.

        Parameters
        ----------
        name : str
            Variable name for the returned batch.
        dtype : str
            Data type for the returned batch.
            If omitted (None), self.dtype is used.
        batch_size : int
            Number of examples in the returned batch.

        Returns
        -------
        batch : TensorVariable, SparseVariable, or tuple thereof
            A batch with the appropriate number of dimensions and
            appropriate broadcast flags to represent a batch of
            points in this space.
        """
        raise NotImplementedError()

    def make_batch_theano(self, name=None, dtype=None, batch_size=None):
        """ An alias to make_theano_batch """

        return self.make_theano_batch(name=name,
                                      dtype=dtype,
                                      batch_size=batch_size)

    @wraps(make_theano_batch)
    def get_theano_batch(self, *args, **kwargs):

        return self.make_theano_batch(*args, **kwargs)

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
        Returns a numeric batch (e.g. a numpy.ndarray or scipy.sparse sparse
        array), formatted to lie in this space.

        This is just a wrapper around self._format_as, with an extra check
        to throw an exception if <batch> is symbolic.

        Should be invertible, i.e. batch should equal
        `space.format_as(self.format_as(batch, space), self)`

        Parameters
        ----------
        batch : numpy.ndarray, or one of the scipy.sparse matrices.
            Array which lies in this space.
        space : Space
            Target space to format batch to.

        Returns
        -------
        WRITEME
            The formatted batch
        """

        self._check_is_numeric(batch)

        return self._format_as(is_numeric=True,
                               batch=batch,
                               space=space)

    def _check_sizes(self, space):
        """
        Called by self._format_as(space), to check whether self and space
        have compatible sizes. Throws a ValueError if they don't.
        """
        my_dimension = self.get_total_dimension()
        other_dimension = space.get_total_dimension()
        if my_dimension != other_dimension:
            raise ValueError(str(self)+" with total dimension " +
                             str(my_dimension) +
                             " can't format a batch into " +
                             str(space) + "because its total dimension is " +
                             str(other_dimension))

    def format_as(self, batch, space):
        """
        .. todo::

            WRITEME
        """
        self._check_is_symbolic(batch)
        return self._format_as(is_numeric=False,
                               batch=batch,
                               space=space)

    def _format_as(self, is_numeric, batch, space):
        """
        The shared implementation of format_as() and np_format_as().
        Agnostic to whether batch is symbolic or numeric, which avoids
        duplicating a lot of code between format_as() and np_format_as().

        Calls the appropriate callbacks, then calls self._format_as_impl().

        Should be invertible, i.e. batch should equal
        `space._format_as(self._format_as(batch, space), self)`

        Parameters
        ----------
        is_numeric : bool
            Set to True to call np_validate_callbacks().
            Set to False to call validate_callbacks().
        batch : WRITEME
        space : Space
            WRITEME

        Returns
        -------
        WRITEME
        """

        assert isinstance(is_numeric, bool)

        # Checks if batch belongs to this space
        self._validate(is_numeric, batch)

        # checks if self and space have compatible sizes for formatting.
        self._check_sizes(space)

        return self._format_as_impl(is_numeric, batch, space)

    def _format_as_impl(self, is_numeric, batch, target_space):
        """
        Actual implementation of format_as/np_format_as. Formats batch to
        target_space.

        Should be invertible, i.e. batch should equal
        `space._format_as_impl(self._format_as_impl(batch, space), self)`

        Parameters
        ----------
        is_numeric : bool
            Set to True to treat batch as a numeric batch, False to
            treat it as a symbolic batch. This is necessary because
            sometimes a batch's numeric/symbolicness can be ambiguous,
            i.e. when it's the empty tuple ().
        batch : a numpy.ndarray, scipy.sparse matrix, theano symbol, or a \
                nested tuple thereof
            Implementations of this method may assume that batch lies in this
            space (i.e. that it passed self._validate(batch) without throwing
            an exception).
        target_space : A Space subclass
            The space to transform batch into.

        Returns
        -------
        The batch, converted to the target_space.
        """

        raise NotImplementedError("%s does not implement _format_as_impl()." %
                                  type(self))

    def validate(self, batch):
        """
        Runs all validate_callbacks, then checks that batch lies in this space.
        Raises an exception if the batch isn't symbolic, or if any of these
        checks fails.

        Parameters
        ----------
        batch : a symbolic (Theano) variable that lies in this space.
        """
        self._check_is_symbolic(batch)
        self._validate(is_numeric=False, batch=batch)

    def np_validate(self, batch):
        """
        Runs all np_validate_callbacks, then checks that batch lies in this
        space. Raises an exception if the batch isn't numeric, or if any of
        these checks fails.

        Parameters
        ----------
        batch : a numeric (numpy/scipy.sparse) variable that lies in this \
                space
        """
        self._check_is_numeric(batch)
        self._validate(is_numeric=True, batch=batch)

    def _validate(self, is_numeric, batch):
        """
        Shared implementation of validate() and np_validate().
        Calls validate_callbacks or np_validate_callbacks as appropriate,
        then calls self._validate_impl(batch) to verify that batch belongs
        to this space.

        Parameters
        ----------
        is_numeric : bool.
            Set to True to call np_validate_callbacks,
            False to call validate_callbacks.
            Necessary because it can be impossible to tell from the
            batch whether it should be treated as a numeric of symbolic
            batch, for example when the batch is the empty tuple (),
			or NullSpace batch None.

        batch : a theano variable, numpy ndarray, scipy.sparse matrix \
                or a nested tuple thereof
            Represents a batch belonging to this space.
        """

        if is_numeric:
            self._check_is_numeric(batch)
            callbacks_name = "np_validate_callbacks"
        else:
            self._check_is_symbolic(batch)
            callbacks_name = "validate_callbacks"

        if not hasattr(self, callbacks_name):
            warnings.warn("It looks like the " + str(type(self)) +
                          "subclass of Space does not call the superclass "
                          "__init__ method. Currently this is a warning. It "
                          "will become an error on or after 2014-06-17.")
        else:
            callbacks = getattr(self, callbacks_name)
            for callback in callbacks:
                callback(batch)

        self._validate_impl(is_numeric, batch)

    def _validate_impl(self, is_numeric, batch):
        """
        Subclasses must override this method so that it throws an
        exception if the batch is the wrong shape or dtype for this Space.

        Parameters
        ----------

        is_numeric : bool
            Set to True to treat batch as a numeric type
            (numpy.ndarray or scipy.sparse matrix).
            Set to False to treat batch as a symbolic (Theano) variable.
            Necessary because batch could be (), which could be numeric
            or symbolic.

        batch : A numpy ndarray, scipy.sparse matrix, theano variable \
                or a nested tuple thereof.
            Must be a valid batch belonging to this space.
        """
        raise NotImplementedError('Class "%s" does not implement '
                                  '_validate_impl()' % type(self))

    def batch_size(self, batch):
        """
        Returns the batch size of a symbolic batch.

        Parameters
        ----------
        batch : WRITEME
        """
        return self._batch_size(is_numeric=False, batch=batch)

    def np_batch_size(self, batch):
        """
        Returns the batch size of a numeric (numpy/scipy.sparse) batch.

        Parameters
        ----------
        batch : WRITEME
        """
        return self._batch_size(is_numeric=True, batch=batch)

    def _batch_size(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        self._validate(is_numeric, batch)
        return self._batch_size_impl(is_numeric, batch)

    def _batch_size_impl(self, is_numeric, batch):
        """
        Returns the batch size of a batch.

        Parameters
        ----------
        batch : WRITEME
        """
        raise NotImplementedError("%s does not implement batch_size" %
                                  type(self))

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

    @staticmethod
    def _check_is_numeric(batch):
        """
        .. todo::

            WRITEME
        """
        if not is_numeric_batch(batch):
            raise TypeError('Expected batch to be a numeric variable, but '
                            'instead it was of type "%s"' % type(batch))

    @staticmethod
    def _check_is_symbolic(batch):
        """
        .. todo::

            WRITEME
        """
        if not is_symbolic_batch(batch):
            raise TypeError('Expected batch to be a symbolic variable, but '
                            'instead it was of type "%s"' % type(batch))

    def _clean_dtype_arg(self, dtype):
        """
        Checks dtype string for validity, and returns it if it is.

        If dtype is 'floatX', returns the theano.config.floatX dtype (this will
        either be 'float32' or 'float64'.
        """

        if isinstance(dtype, np.dtype):
            dtype = str(dtype)

        if dtype == 'floatX':
            return theano.config.floatX

        if dtype is None or \
           dtype in tuple(x.dtype for x in theano.scalar.all_types):
            return dtype

        raise TypeError('Unrecognized value "%s" (type %s) for dtype arg' %
                        (dtype, type(dtype)))


class SimplyTypedSpace(Space):
    """
    An abstract base class for Spaces that use a numpy/theano dtype string for
    its .dtype property.
    """

    def __init__(self, dtype='floatX', **kwargs):
        super(SimplyTypedSpace, self).__init__(**kwargs)
        self._dtype = super(SimplyTypedSpace, self)._clean_dtype_arg(dtype)

    def _clean_dtype_arg(self, dtype):
        """
        if dtype is None, checks that self.dtype is not None.
        Otherwise, same as superclass' implementation.
        """

        if dtype is None:
            if self.dtype is None:
                raise TypeError("self.dtype is None, so you must provide a "
                                "non-None dtype argument to this method.")
            return self.dtype

        return super(SimplyTypedSpace, self)._clean_dtype_arg(dtype)

    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        if isinstance(batch, tuple):
            raise TypeError("This space only supports simple dtypes, but "
                            "received a composite batch.")

        def is_complex(dtype):
            return str(dtype).startswith('complex')

        if self.dtype is not None and \
           is_complex(batch.dtype) and \
           not is_complex(self.dtype):
            raise TypeError("This space has a non-complex dtype (%s), and "
                            "thus cannot support complex batches of type %s."
                            % (self.dtype, batch.dtype))

    @property
    def dtype(self):
        """
        .. todo::

            WRITEME
        """
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype):
        """
        .. todo::

            WRITEME
        """
        self._dtype = super(SimplyTypedSpace, self)._clean_dtype_arg(new_dtype)

    def __setstate__(self, state_dict):
        """
        .. todo::

            WRITEME
        """
        self.__dict__.update(state_dict)

        # When unpickling a Space that was pickled before Spaces had dtypes,
        # we need to set the _dtype to the default value.
        if not '_dtype' in state_dict:
            self._dtype = theano.config.floatX


class IndexSpace(SimplyTypedSpace):
    """
    A space representing indices, for example MNIST labels (0-10) or the
    indices of words in a dictionary for NLP tasks. A single space can
    contain multiple indices, for example the word indices of an n-gram.

    IndexSpaces can be converted to VectorSpaces in two ways: Either the
    labels are converted into one-hot vectors which are then concatenated,
    or they are converted into a single vector where 1s indicate labels
    present i.e. for 4 possible labels we have [0, 2] -> [1 0 1 0] or
    [0, 2] -> [1 0 0 0 0 0 1 0].

    Parameters
    ----------
    max_labels : int
        The number of possible classes/labels. This means that
        all labels should be < max_labels. Example: For MNIST
        there are 10 numbers and hence max_labels = 10.
    dim : int
        The number of indices in one space e.g. for MNIST there is
        one target label and hence dim = 1. If we have an n-gram
        of word indices as input to a neurel net language model, dim = n.
    dtype : str
        A numpy dtype string indicating this space's dtype.
        Must be an integer type e.g. int32 or int64.
    kwargs : dict
        Passes on to superclass constructor
    """
    def __init__(self, max_labels, dim, dtype='int64', **kwargs):
        if not 'int' in dtype:
            raise ValueError("The dtype of IndexSpace must be an integer type")

        super(IndexSpace, self).__init__(dtype, **kwargs)

        self.max_labels = max_labels
        self.dim = dim
        self.formatter = OneHotFormatter(self.max_labels)

    def __str__(self):
        """Return a string representation"""
        return ('%(classname)s(dim=%(dim)s, max_labels=%(max_labels)s, '
                'dtype=%(dtype)s)') % dict(classname=self.__class__.__name__,
                                           dim=self.dim,
                                           max_labels=self.max_labels,
                                           dtype=self.dtype)

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and
                self.max_labels == other.max_labels and
                self.dim == other.dim and
                self.dtype == other.dtype)

    def __ne__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (not self == other)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.dim

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((1, self.dim,))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)
        return np.zeros((batch_size, self.dim), dtype=dtype)

    @functools.wraps(Space._check_sizes)
    def _check_sizes(self, space):
        if isinstance(space, VectorSpace):
            if space.dim not in (self.max_labels,              # merged onehots
                                 self.dim * self.max_labels):  # concatenated
                raise ValueError("Can't convert to VectorSpace of dim %d. "
                                 "Expected either dim=%d (merged one-hots) or "
                                 "%d (concatenated one-hots)" %
                                 (space.dim,
                                  self.max_labels,
                                  self.dim * self.max_labels))
        elif isinstance(space, IndexSpace):
            if space.dim != self.dim or space.max_labels != self.max_labels:
                raise ValueError("Can't convert to IndexSpace of dim %d and "
                                 "max_labels %d." %
                                 (space.dim, self.max_labels))
        else:
            raise ValueError("Can't convert to " + str(space.__class__))

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSpace):
            if self.max_labels == space.dim:
                mode = 'merge'
            elif self.dim * self.max_labels == space.dim:
                mode = 'concatenate'
            else:
                raise ValueError("There is a bug. Couldn't format to a "
                                 "VectorSpace because it had an incorrect "
                                 "size, but this should've been caught in "
                                 "IndexSpace._check_sizes().")

            format_func = (self.formatter.format if is_numeric else
                           self.formatter.theano_expr)
            return _cast(format_func(batch, sparse=space.sparse, mode=mode),
                         space.dtype)
        elif isinstance(space, IndexSpace):
            if space.dim != self.dim or space.max_labels != self.max_labels:
                raise ValueError("The two IndexSpaces' dim and max_labels "
                                 "values don't match. This should have been "
                                 "caught by IndexSpace._check_sizes().")

            return _cast(batch, space.dtype)
        else:
            raise ValueError("Can't convert %s to %s"
                             % (self, space))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if batch_size == 1:
            rval = tensor.lrow(name=name)
        else:
            rval = tensor.lmatrix(name=name)

        if theano.config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)
        return rval

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[0]

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(IndexSpace, self)._validate_impl(is_numeric, batch)

        if is_numeric:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a IndexSpace batch should be a "
                                "numpy.ndarray, or CudaNdarray, but is %s."
                                % str(type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a IndexSpace batch must be "
                                 "2D, got %d dimensions for %s." % (batch.ndim,
                                                                    batch))
            if batch.shape[1] != self.dim:
                raise ValueError("The width of a IndexSpace batch must match "
                                 "with the space's dimension, but batch has "
                                 "shape %s and dim = %d." % (str(batch.shape),
                                                             self.dim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("IndexSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("IndexSpace batch should be TensorType or "
                                "CudaNdarrayType, got "+str(batch.type))
            if batch.ndim != 2:
                raise ValueError('IndexSpace batches must be 2D, got %d '
                                 'dimensions' % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)


class VectorSpace(SimplyTypedSpace):
    """
    A space whose points are defined as fixed-length vectors.

    Parameters
    ----------
    dim : int
        Dimensionality of a vector in this space.
    sparse : bool, optional
        Sparse vector or not
    dtype : str, optional
        A numpy dtype string (e.g. 'float32') indicating this space's
        dtype, or None for a dtype-agnostic space.
    kwargs : dict
        Passed on to superclass constructor.
    """

    def __init__(self,
                 dim,
                 sparse=False,
                 dtype='floatX',
                 **kwargs):
        super(VectorSpace, self).__init__(dtype, **kwargs)

        self.dim = dim
        self.sparse = sparse

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return ('%s(dim=%d%s, dtype=%s)' %
                (self.__class__.__name__,
                 self.dim,
                 ', sparse' if self.sparse else '',
                 self.dtype))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((self.dim,))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)

        if self.sparse:
            return scipy.sparse.csr_matrix((batch_size, self.dim), dtype=dtype)
        else:
            return np.zeros((batch_size, self.dim), dtype=dtype)

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[0]

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        dtype = self._clean_dtype_arg(dtype)

        if self.sparse:
            if batch_size is not None:
                raise NotImplementedError("batch_size not implemented "
                                          "for sparse case")
            rval = theano.sparse.csr_matrix(name=name, dtype=dtype)
        else:
            if batch_size == 1:
                rval = tensor.row(name=name, dtype=dtype)
            else:
                rval = tensor.matrix(name=name, dtype=dtype)

        if theano.config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)
        return rval

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.dim

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        to_type = None

        def is_sparse(batch):
            return (isinstance(batch, theano.sparse.SparseVariable) or
                    scipy.sparse.issparse(batch))

        if not isinstance(space, IndexSpace):
            my_dimension = self.get_total_dimension()
            other_dimension = space.get_total_dimension()
            if my_dimension != other_dimension:
                raise ValueError(str(self)+" with total dimension " +
                                 str(my_dimension) +
                                 " can't format a batch into " +
                                 str(space) +
                                 "because its total dimension is " +
                                 str(other_dimension))

        if isinstance(space, CompositeSpace):
            if isinstance(batch, theano.sparse.SparseVariable):
                warnings.warn('Formatting from a sparse VectorSpace to a '
                              'CompositeSpace is currently (2 Jan 2014) a '
                              'non-differentiable action. This is because it '
                              'calls slicing operations on a sparse batch '
                              '(e.g. "my_matrix[r:R, c:C]", which Theano does '
                              'not yet have a gradient operator for. If '
                              'autodifferentiation is reporting an error, '
                              'this may be why. Formatting batch type %s '
                              'from space %s to space %s' %
                              (type(batch), self, space))
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:, pos:pos+width]
                pos += width
                vector_subspace = VectorSpace(dim=width,
                                              dtype=self.dtype,
                                              sparse=self.sparse)
                formatted = vector_subspace._format_as(is_numeric,
                                                       subtensor,
                                                       component)
                pieces.append(formatted)

            result = tuple(pieces)

        elif isinstance(space, Conv2DSpace):
            if is_sparse(batch):
                raise TypeError("Formatting a SparseVariable to a Conv2DSpace "
                                "is not supported, since neither scipy nor "
                                "Theano has sparse tensors with more than 2 "
                                "dimensions. We need 4 dimensions to "
                                "represent a Conv2DSpace batch")

            dims = {'b': batch.shape[0],
                    'c': space.num_channels,
                    0: space.shape[0],
                    1: space.shape[1]}
            if space.axes != space.default_axes:
                # Always use default_axes, so conversions like
                # Conv2DSpace(c01b) -> VectorSpace -> Conv2DSpace(b01c) work
                shape = [dims[ax] for ax in space.default_axes]
                batch = _reshape(batch, shape)
                batch = batch.transpose(*[space.default_axes.index(ax)
                                          for ax in space.axes])
                result = batch
            else:
                shape = tuple([dims[elem] for elem in space.axes])
                result = _reshape(batch, shape)

            to_type = space.dtype

        elif isinstance(space, VectorSpace):
            if self.dim != space.dim:
                raise ValueError("Can't convert between VectorSpaces of "
                                 "different sizes (%d to %d)."
                                 % (self.dim, space.dim))

            if space.sparse != is_sparse(batch):
                if space.sparse:
                    batch = _dense_to_sparse(batch)
                elif isinstance(batch, theano.sparse.SparseVariable):
                    batch = theano.sparse.dense_from_sparse(batch)
                elif scipy.sparse.issparse(batch):
                    batch = batch.todense()
                else:
                    assert False, ("Unplanned-for branch in if-elif-elif "
                                   "chain. This is a bug in the code.")

            result = batch
            to_type = space.dtype
        else:
            raise NotImplementedError("%s doesn't know how to format as %s" %
                                      (self, space))

        return _cast(result, dtype=to_type)

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and
                self.dim == other.dim and
                self.sparse == other.sparse and
                self.dtype == other.dtype)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), self.dim, self.sparse, self.dtype))

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """

        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(VectorSpace, self)._validate_impl(is_numeric, batch)

        if isinstance(batch, theano.gof.Variable):
            if self.sparse:
                if not isinstance(batch.type, theano.sparse.SparseType):
                    raise TypeError('This VectorSpace is%s sparse, but the '
                                    'provided batch is not. (batch type: "%s")'
                                    % ('' if self.sparse else ' not',
                                       type(batch)))
            elif not isinstance(batch.type, (theano.tensor.TensorType,
                                             CudaNdarrayType)):
                raise TypeError("VectorSpace batch should be TensorType or "
                                "CudaNdarrayType, got "+str(batch.type))

            if batch.ndim != 2:
                raise ValueError('VectorSpace batches must be 2D, got %d '
                                 'dimensions' % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)  # sic; val is numeric, not symbolic
        else:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if (not self.sparse
                    and not isinstance(batch, np.ndarray)
                    and type(batch) != 'CudaNdarray'):
                raise TypeError("The value of a VectorSpace batch should be a "
                                "numpy.ndarray, or CudaNdarray, but is %s."
                                % str(type(batch)))
            if self.sparse:
                if not theano.sparse.enable_sparse:
                    raise TypeError("theano.sparse is not enabled, cannot "
                                    "have a value for a sparse VectorSpace.")
                if not scipy.sparse.issparse(batch):
                    raise TypeError("The value of a sparse VectorSpace batch "
                                    "should be a sparse scipy matrix, got %s "
                                    "of type %s." % (batch, type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a VectorSpace batch must be "
                                 "2D, got %d dimensions for %s." % (batch.ndim,
                                                                    batch))
            if batch.shape[1] != self.dim:
                raise ValueError("The width of a VectorSpace batch must match "
                                 "with the space's dimension, but batch has "
                                 "shape %s and dim = %d." %
                                 (str(batch.shape), self.dim))


class VectorSequenceSpace(SimplyTypedSpace):
    """
    A space representing a single, variable-length sequence of fixed-sized
    vectors.

    Parameters
    ----------
    dim : int
        Vector size
    dtype : str, optional
        A numpy dtype string indicating this space's dtype.
    kwargs : dict
        Passes on to superclass constructor
    """
    def __init__(self, dim, dtype='floatX', **kwargs):
        super(VectorSequenceSpace, self).__init__(dtype, **kwargs)
        self.dim = dim

    def __str__(self):
        """Return a string representation"""
        return ('%(classname)s(dim=%(dim)s, dtype=%(dtype)s)' %
                dict(classname=self.__class__.__name__,
                     dim=self.dim,
                     dtype=self.dtype))

    @wraps(Space.__eq__)
    def __eq__(self, other):
        return (type(self) == type(other) and
                self.dim == other.dim and
                self.dtype == other.dtype)

    @wraps(Space._check_sizes)
    def _check_sizes(self, space):
        if not isinstance(space, VectorSequenceSpace):
            raise ValueError("Can't convert to " + str(space.__class__))
        else:
            if space.dim != self.dim:
                raise ValueError("Can't convert to VectorSequenceSpace of "
                                 "dim %d" %
                                 (space.dim,))

    @wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSequenceSpace):
            if space.dim != self.dim:
                raise ValueError("The two VectorSequenceSpaces' dim "
                                 "values don't match. This should have been "
                                 "caught by "
                                 "VectorSequenceSpace._check_sizes().")

            return _cast(batch, space.dtype)
        else:
            raise ValueError("Can't convert %s to %s" % (self, space))

    @wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if batch_size == 1:
            return tensor.matrix(name=name)
        else:
            return ValueError("VectorSequenceSpace does not support batches "
                              "of sequences.")

    @wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        # Only batch size of 1 is supported
        return 1

    @wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(VectorSequenceSpace, self)._validate_impl(is_numeric, batch)

        if is_numeric:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a VectorSequenceSpace batch "
                                "should be a numpy.ndarray, or CudaNdarray, "
                                "but is %s." % str(type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a VectorSequenceSpace batch "
                                 "must be 2D, got %d dimensions for %s."
                                 % (batch.ndim, batch))
            if batch.shape[1] != self.dim:
                raise ValueError("The width of a VectorSequenceSpace 'batch' "
                                 "must match with the space's window"
                                 "dimension, but batch has dim %d and "
                                 "this space's dim is %d."
                                 % (batch.shape[1], self.dim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("VectorSequenceSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("VectorSequenceSpace batch should be "
                                "TensorType or CudaNdarrayType, got " +
                                str(batch.type))
            if batch.ndim != 2:
                raise ValueError("VectorSequenceSpace 'batches' must be 2D, "
                                 "got %d dimensions" % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)


class IndexSequenceSpace(SimplyTypedSpace):
    """
    A space representing a single, variable-length sequence of indexes.

    Parameters
    ----------
    max_labels : int
        The number of possible classes/labels. This means that
        all labels should be < max_labels.
    dim : int
        The number of indices in one element of the sequence
    dtype : str
        A numpy dtype string indicating this space's dtype.
        Must be an integer type e.g. int32 or int64.
    kwargs : dict
        Passes on to superclass constructor
    """
    def __init__(self, max_labels, dim, dtype='int64', **kwargs):
        if not 'int' in dtype:
            raise ValueError("The dtype of IndexSequenceSpace must be an "
                             "integer type")

        super(IndexSequenceSpace, self).__init__(dtype, **kwargs)

        self.max_labels = max_labels
        self.dim = dim
        self.formatter = OneHotFormatter(self.max_labels)

    def __str__(self):
        """Return a string representation"""
        return ('%(classname)s(dim=%(dim)s, max_labels=%(max_labels)s, '
                'dtype=%(dtype)s)') % dict(classname=self.__class__.__name__,
                                           dim=self.dim,
                                           max_labels=self.max_labels,
                                           dtype=self.dtype)

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and
                self.max_labels == other.max_labels and
                self.dim == other.dim and
                self.dtype == other.dtype)

    @wraps(Space._check_sizes)
    def _check_sizes(self, space):
        if isinstance(space, VectorSequenceSpace):
            # self.max_labels -> merged onehots
            # self.dim * self.max_labels -> concatenated
            if space.dim not in (self.max_labels, self.dim * self.max_labels):
                raise ValueError("Can't convert to VectorSequenceSpace of "
                                 "dim %d. Expected either "
                                 "dim=%d (merged one-hots) or %d "
                                 "(concatenated one-hots)" %
                                 (space.dim,
                                  self.max_labels,
                                  self.dim * self.max_labels))
        elif isinstance(space, IndexSequenceSpace):
            if space.dim != self.dim or space.max_labels != self.max_labels:
                raise ValueError("Can't convert to IndexSequenceSpace of "
                                 "dim %d and max_labels %d." %
                                 (space.dim, self.max_labels))
        else:
            raise ValueError("Can't convert to " + str(space.__class__))

    @wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSequenceSpace):
            if self.max_labels == space.dim:
                mode = 'merge'
            elif self.dim * self.max_labels == space.dim:
                mode = 'concatenate'
            else:
                raise ValueError("There is a bug. Couldn't format to a "
                                 "VectorSequenceSpace because it had an "
                                 "incorrect size, but this should've been "
                                 "caught in "
                                 "IndexSequenceSpace._check_sizes().")

            format_func = (self.formatter.format if is_numeric else
                           self.formatter.theano_expr)
            return _cast(format_func(batch, mode=mode), space.dtype)
        elif isinstance(space, IndexSequenceSpace):
            if space.dim != self.dim or space.max_labels != self.max_labels:
                raise ValueError("The two IndexSequenceSpaces' dim and "
                                 "max_labels values don't match. This should "
                                 "have been caught by "
                                 "IndexSequenceSpace._check_sizes().")

            return _cast(batch, space.dtype)
        else:
            raise ValueError("Can't convert %s to %s"
                             % (self, space))

    @wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if batch_size == 1:
            return tensor.matrix(name=name)
        else:
            return ValueError("IndexSequenceSpace does not support batches "
                              "of sequences.")

    @wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        # Only batch size of 1 is supported
        return 1

    @wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(IndexSequenceSpace, self)._validate_impl(is_numeric, batch)

        if is_numeric:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a IndexSequenceSpace batch "
                                "should be a numpy.ndarray, or CudaNdarray, "
                                "but is %s." % str(type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a IndexSequenceSpace batch "
                                 "must be 2D, got %d dimensions for %s." %
                                 (batch.ndim, batch))
            if batch.shape[1] != self.dim:
                raise ValueError("The width of a IndexSequenceSpace batch "
                                 "must match with the space's dimension, but "
                                 "batch has shape %s and dim = %d." %
                                 (str(batch.shape), self.dim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("IndexSequenceSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("IndexSequenceSpace batch should be "
                                "TensorType or CudaNdarrayType, got " +
                                str(batch.type))
            if batch.ndim != 2:
                raise ValueError('IndexSequenceSpace batches must be 2D, got '
                                 '%d dimensions' % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)


class Conv2DSpace(SimplyTypedSpace):
    """
    A space whose points are 3-D tensors representing (potentially
    multi-channel) images.

    Parameters
    ----------
    shape : sequence, length 2
        The shape of a single image, i.e. (rows, cols).
    num_channels : int (synonym: channels)
        Number of channels in the image, i.e. 3 if RGB.
    axes : tuple
        A tuple indicating the semantics of each axis, containing the
        following elements in some order:

            - 'b' : this axis is the batch index of a minibatch.
            - 'c' : this axis the channel index of a minibatch.
            - 0 : topological axis 0 (rows)
            - 1 : topological axis 1 (columns)

        For example, a PIL image has axes (0, 1, 'c') or (0, 1).
        The pylearn2 image displaying functionality uses
        ('b', 0, 1, 'c') for batches and (0, 1, 'c') for images.
        theano's conv2d operator uses ('b', 'c', 0, 1) images.
    dtype : str
        A numpy dtype string (e.g. 'float32') indicating this space's
        dtype, or None for a dtype-agnostic space.
    kwargs : dict
        Passed on to superclass constructor
    """


    # Assume pylearn2's get_topological_view format, since this is how
    # data is currently served up. If we make better iterators change
    # default to ('b', 'c', 0, 1) for theano conv2d
    default_axes = ('b', 0, 1, 'c')

    def __init__(self,
                 shape,
                 channels=None,
                 num_channels=None,
                 axes=None,
                 dtype='floatX',
                 **kwargs):

        super(Conv2DSpace, self).__init__(dtype, **kwargs)

        assert (channels is None) + (num_channels is None) == 1
        if num_channels is None:
            num_channels = channels

        assert isinstance(num_channels, py_integer_types)

        if not hasattr(shape, '__len__'):
            raise ValueError("shape argument for Conv2DSpace must have a "
                             "length. Got %s." % str(shape))

        if len(shape) != 2:
            raise ValueError("shape argument to Conv2DSpace must be length 2, "
                             "not %d" % len(shape))

        assert all(isinstance(elem, py_integer_types) for elem in shape)
        assert all(elem > 0 for elem in shape)
        assert isinstance(num_channels, py_integer_types)
        assert num_channels > 0
        # Converts shape to a tuple, so it can be hashable, and self can be too
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
        return ("%s(shape=%s, num_channels=%d, axes=%s, dtype=%s)" %
                (self.__class__.__name__,
                 str(self.shape),
                 self.num_channels,
                 str(self.axes),
                 self.dtype))

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        assert isinstance(self.axes, tuple)

        if isinstance(other, Conv2DSpace):
            assert isinstance(other.axes, tuple)

        return (type(self) == type(other) and
                self.shape == other.shape and
                self.num_channels == other.num_channels and
                self.axes == other.axes and
                self.dtype == other.dtype)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self),
                     self.shape,
                     self.num_channels,
                     self.axes,
                     self.dtype))

    @functools.wraps(Space.get_batch_axis)
    def get_batch_axis(self):
        return self.axes.index('b')

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = {0: self.shape[0], 1: self.shape[1], 'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes if elem != 'b']
        return np.zeros(shape, dtype=self.dtype)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)

        if not isinstance(batch_size, py_integer_types):
            raise TypeError("Conv2DSpace.get_origin_batch expects an int, "
                            "got %s of type %s" % (str(batch_size),
                                                   type(batch_size)))
        assert batch_size > 0
        dims = {'b': batch_size,
                0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes]
        return np.zeros(shape, dtype=dtype)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        dtype = self._clean_dtype_arg(dtype)
        broadcastable = [False] * 4
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = (batch_size == 1)
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable
                          )(name=name)
        if theano.config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)
        return rval

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
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

        if is_symbolic_batch(tensor):
            return tensor.dimshuffle(*shuffle)
        else:
            return tensor.transpose(*shuffle)

    @staticmethod
    def convert_numpy(tensor, src_axes, dst_axes):
        """
        .. todo::

            WRITEME
        """
        return Conv2DSpace.convert(tensor, src_axes, dst_axes)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):

        # Patch old pickle files
        if not hasattr(self, 'num_channels'):
            self.num_channels = self.nchannels

        return self.shape[0] * self.shape[1] * self.num_channels

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # checks batch.type against self.dtype
        super(Conv2DSpace, self)._validate_impl(is_numeric, batch)

        if isinstance(batch, theano.gof.Variable):
            if isinstance(batch, theano.sparse.SparseVariable):
                raise TypeError("Conv2DSpace cannot use SparseVariables, "
                                "since as of this writing (28 Dec 2013), "
                                "there is not yet a SparseVariable type with "
                                "4 dimensions")

            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("Conv2DSpace batches must be theano "
                                "Variables, got " + str(type(batch)))

            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError('Expected TensorType or CudaNdArrayType, got '
                                '"%s"' % type(batch.type))

            if batch.ndim != 4:
                raise ValueError("The value of a Conv2DSpace batch must be "
                                 "4D, got %d dimensions for %s." %
                                 (batch.ndim, batch))

            for val in get_debug_values(batch):
                self.np_validate(val)
        else:
            if scipy.sparse.issparse(batch):
                raise TypeError("Conv2DSpace cannot use sparse batches, since "
                                "scipy.sparse does not support 4 dimensional "
                                "tensors currently (28 Dec 2013).")

            if (not isinstance(batch, np.ndarray)) \
               and type(batch) != 'CudaNdarray':
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
                raise ValueError("Expected axis %d to be number of channels "
                                 "(%d) but it is %d" %
                                 (d, self.num_channels, actual_channels))
            assert batch.shape[self.axes.index('c')] == self.num_channels

            for coord in [0, 1]:
                d = self.axes.index(coord)
                actual_shape = batch.shape[d]
                expected_shape = self.shape[coord]
                if actual_shape != expected_shape:
                    raise ValueError("Conv2DSpace with shape %s and axes %s "
                                     "expected dimension %s of a batch (%s) "
                                     "to have length %s but it has %s"
                                     % (str(self.shape),
                                        str(self.axes),
                                        str(d),
                                        str(batch),
                                        str(expected_shape),
                                        str(actual_shape)))

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSpace):
            # We need to ensure that the resulting batch will always be
            # the same in `space`, no matter what the axes of `self` are.
            if self.axes != self.default_axes:
                # The batch index goes on the first axis
                assert self.default_axes[0] == 'b'
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in self.default_axes])
            result = batch.reshape((batch.shape[0],
                                    self.get_total_dimension()))
            if space.sparse:
                result = _dense_to_sparse(result)

        elif isinstance(space, Conv2DSpace):
            result = Conv2DSpace.convert(batch, self.axes, space.axes)
        else:
            raise NotImplementedError("%s doesn't know how to format as %s"
                                      % (str(self), str(space)))

        return _cast(result, space.dtype)


class CompositeSpace(Space):
    """
    A Space whose points are tuples of points in other spaces.
    May be nested, in which case the points are nested tuples.

    Parameters
    ----------
    components : WRITEME
    kwargs : dict
        WRITEME
    """
    def __init__(self, components, **kwargs):
        super(CompositeSpace, self).__init__(**kwargs)

        assert isinstance(components, (list, tuple))

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
                all(my_component == other_component for
                    my_component, other_component in
                    zip(self.components, other.components)))

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

    @property
    def dtype(self):
        """
        Returns a nested tuple of dtype strings. NullSpaces will yield a bogus
        dtype string (see NullSpace.dtype).
        """

        def get_dtype_of_space(space):
            if isinstance(space, CompositeSpace):
                return tuple(get_dtype_of_space(c) for c in space.components)
            elif isinstance(space, NullSpace):
                return NullSpace().dtype
            else:
                return space.dtype

        return get_dtype_of_space(self)

    @dtype.setter
    def dtype(self, new_dtype):
        """
        If new_dtype is None or a string, it will be applied to all components
        (except any NullSpaces).

        If new_dtype is a (nested) tuple, its elements will be applied to
        corresponding components.
        """
        if isinstance(new_dtype, tuple):
            for component, new_dt in safe_zip(self.components, new_dtype):
                component.dtype = new_dt
        elif new_dtype is None or isinstance(new_dtype, str):
            for component in self.components:
                if not isinstance(component, NullSpace):
                    component.dtype = new_dtype

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
        present in subset.

        May not be a tuple anymore if there is only one index.
        Outputs will be ordered in the order that they appear in subset.

        Only supports symbolic batches.

        Parameters
        ----------
        batch : WRITEME
        subset : WRITEME
        """

        self._validate(is_numeric=False, batch=batch)
        assert isinstance(subset, (list, tuple))

        if len(subset) == 1:
            idx, = subset
            return batch[idx]

        return tuple([batch[i] for i in subset])

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return sum([component.get_total_dimension() for component in
                    self.components])

    @functools.wraps(Space.make_shared_batch)
    def make_shared_batch(self, batch_size, name=None, dtype=None):
        dtype = self._clean_dtype_arg(dtype)
        batch = self.get_origin_batch(batch_size, dtype)

        def recursive_shared(batch):
            if isinstance(batch, tuple):
                return tuple(recursive_shared(b) for b in batch)
            else:
                return theano.shared(batch, name=name)

        return recursive_shared(batch)

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
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

              `self_component._format_as(is_numeric,
                                         batch_component,
                                         space_component)`

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
                subspace = VectorSpace(dim=component.get_total_dimension(),
                                       dtype=space.dtype,
                                       sparse=space.sparse)
                pieces.append(component._format_as(is_numeric,
                                                   input_piece,
                                                   subspace))

            # Pieces should all have the same dtype, before we concatenate them
            if len(pieces) > 0:
                for piece in pieces[1:]:
                    if pieces[0].dtype != piece.dtype:
                        assert space.dtype is None
                        raise TypeError("Tried to format components with "
                                        "differing dtypes into a VectorSpace "
                                        "with no dtype of its own. "
                                        "dtypes: %s" %
                                        str(tuple(str(p.dtype)
                                                  for p in pieces)))

            if is_symbolic_batch(batch):
                if space.sparse:
                    return theano.sparse.hstack(pieces)
                else:
                    return tensor.concatenate(pieces, axis=1)
            else:
                if space.sparse:
                    return scipy.sparse.hstack(pieces)
                else:
                    return np.concatenate(pieces, axis=1)

        if isinstance(space, CompositeSpace):
            def recursive_format_as(orig_space, batch, dest_space):
                if not (isinstance(orig_space, CompositeSpace) ==
                        isinstance(dest_space, CompositeSpace)):
                    raise TypeError("Can't convert between CompositeSpaces "
                                    "with different tree structures")

                # No need to check batch's tree structure. Space._format_as()
                # already did that by calling _validate(), before calling this
                # method.

                if isinstance(orig_space, CompositeSpace):
                    return tuple(recursive_format_as(os, bt, ds)
                                 for os, bt, ds
                                 in safe_zip(orig_space.components,
                                             batch,
                                             dest_space.components))
                else:
                    return orig_space._format_as(is_numeric, batch, dest_space)

            return recursive_format_as(self, batch, space)

        raise NotImplementedError(str(self) +
                                  " does not know how to format as " +
                                  str(space))

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        if not isinstance(batch, tuple):
            raise TypeError("The value of a CompositeSpace batch should be a "
                            "tuple, but is %s of type %s." %
                            (batch, type(batch)))
        if len(batch) != len(self.components):
            raise ValueError("Expected %d elements in batch, got %d"
                             % (len(self.components), len(batch)))
        for batch_elem, component in zip(batch, self.components):
            component._validate(is_numeric, batch_elem)

    def get_origin_batch(self, batch_size, dtype=None):
        """
        Calls get_origin_batch on all subspaces, and returns a (nested)
        tuple containing their return values.

        Parameters
        ----------
        batch_size : int
            Batch size.
        dtype : str
            the dtype to use for all the get_origin_batch() calls on
            subspaces. If dtype is None, or a single dtype string, that will
            be used for all calls. If dtype is a (nested) tuple, it must
            mirror the tree structure of this CompositeSpace.
        """

        dtype = self._clean_dtype_arg(dtype)

        return tuple(component.get_origin_batch(batch_size, dt)
                     for component, dt
                     in safe_zip(self.components, dtype))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self,
                          name=None,
                          dtype=None,
                          batch_size=None):
        """
        Calls make_theano_batch on all subspaces, and returns a (nested)
        tuple containing their return values.

        Parameters
        ----------
        name : str
            Name of the symbolic variable
        dtype : str
            The dtype of the returned batch.
            If dtype is a string, it will be applied to all components.
            If dtype is None, C.dtype will be used for each component C.
            If dtype is a nested tuple, its elements will be applied to
            corresponding elements in the components.
        batch_size : int
            Batch size.
        """

        if name is None:
            name = [None] * len(self.components)
        elif not isinstance(name, (list, tuple)):
            name = ['%s[%i]' % (name, i) for i in xrange(len(self.components))]

        dtype = self._clean_dtype_arg(dtype)

        assert isinstance(name, (list, tuple))
        assert isinstance(dtype, (list, tuple))

        rval = tuple([x.make_theano_batch(name=n,
                                          dtype=d,
                                          batch_size=batch_size)
                      for x, n, d in safe_zip(self.components,
                                              name,
                                              dtype)])
        return rval

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):

        def has_no_data(space):
            """
            Returns True if space can contain no data.
            """
            return (isinstance(subspace, NullSpace) or
                    (isinstance(subspace, CompositeSpace) and
                     len(subspace.components) == 0))

        if is_symbolic_batch(batch):
            for subspace, subbatch in safe_zip(self.components, batch):
                if not has_no_data(subspace):
                    return subspace._batch_size(is_numeric, subbatch)

            return 0  # TODO: shouldn't this line return a Theano object?
        else:
            result = None
            for subspace, subbatch in safe_zip(self.components, batch):
                batch_size = subspace._batch_size(is_numeric, subbatch)
                if has_no_data(subspace):
                    assert batch_size == 0
                else:
                    if result is None:
                        result = batch_size
                    elif batch_size != result:
                        raise ValueError("All non-empty components of a "
                                         "CompositeSpace should have the same "
                                         "batch size, but we encountered "
                                         "components with size %s, then %s." %
                                         (result, batch_size))

            return 0 if result is None else result

    def _clean_dtype_arg(self, dtype):
        """
        If dtype is None or a string, this returns a nested tuple that mirrors
        the tree structure of this CompositeSpace, with dtype at the leaves.

        If dtype is a nested tuple, this checks that it has the same tree
        structure as this CompositeSpace.
        """
        super_self = super(CompositeSpace, self)

        def make_dtype_tree(dtype, space):
            """
            Creates a nested tuple tree that mirrors the tree structure of
            <space>, populating the leaves with <dtype>.
            """
            if isinstance(space, CompositeSpace):
                return tuple(make_dtype_tree(dtype, component)
                             for component in space.components)
            else:
                return super_self._clean_dtype_arg(dtype)

        def check_dtype_tree(dtype, space):
            """
            Verifies that a dtype tree mirrors the tree structure of <space>,
            calling Space._clean_dtype_arg on the leaves.
            """
            if isinstance(space, CompositeSpace):
                if not isinstance(dtype, tuple):
                    raise TypeError("Tree structure mismatch.")

                return tuple(check_dtype_tree(dt, c)
                             for dt, c in safe_zip(dtype, space.components))
            else:
                if not (dtype is None or isinstance(dtype, str)):
                    raise TypeError("Tree structure mismatch.")

                return super_self._clean_dtype_arg(dtype)

        if dtype is None or isinstance(dtype, str):
            dtype = super_self._clean_dtype_arg(dtype)
            return make_dtype_tree(dtype, self)
        else:
            return check_dtype_tree(dtype, self)


class NullSpace(Space):
    """
    A space that contains no data. As such, it has the following quirks:

    * Its validate()/np_validate() methods only accept None.
    * Its dtype string is "Nullspace's dtype".
    * The source name associated to this Space is the empty string ('').
    """

    # NullSpaces don't support validation callbacks, since they only take None
    # as data batches.
    def __init__(self):
        super(NullSpace, self).__init__()

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

    @property
    def dtype(self):
        """
        .. todo::

            WRITEME
        """
        return "%s's dtype" % self.__class__.__name__

    @dtype.setter
    def dtype(self, new_dtype):
        """
        .. todo::

            WRITEME
        """
        if new_dtype != self.dtype:
            raise TypeError('%s can only take the bogus dtype "%s"' %
                            (self.__class__.__name__,
                             self.dtype))

        # otherwise, do nothing

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None):
        return None

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        if batch is not None:
            raise TypeError('NullSpace only accepts None, as a dummy data '
                            'batch. Instead, got %s of type %s'
                            % (batch, type(batch)))

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        assert isinstance(space, NullSpace)
        return None

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        # There is no way to know how many examples would actually
        # have been in the batch, since it is empty. We return 0.
        self._validate(is_numeric, batch)
        return 0
