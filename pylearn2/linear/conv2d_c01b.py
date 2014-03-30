"""
The functionality in this module is very similar to that in
pylearn2.linear.conv2d. The difference is that this module is
based on Alex Krizhevsky's cuda-convnet convolution, while
pylearn2.linear.conv2d is based on theano's 2D convolution.
This module therefore uses the axis format ('c', 0, 1, 'b')
as its native format, while the other uses ('b', 'c', 0, 1).
This module also requires the use of GPU, while the other
supports CPU.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import functools
import logging
import numpy as np
import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox import cuda
import theano.tensor as T

if cuda.cuda_available:
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    from theano.sandbox.cuda import gpu_from_host
    from theano.sandbox.cuda import host_from_gpu

from pylearn2.linear.conv2d import default_seed, default_sparse_seed
from pylearn2.linear.linear_transform import LinearTransform
from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.filter_acts import ImageActs
from pylearn2.space import Conv2DSpace
from pylearn2.utils.call_check import checked_call
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


class Conv2D(LinearTransform):
    """
    A pylearn2 linear operator based on 2D convolution,
    implemented using Alex Krizhevsky's cuda-convnet library.

    Parameters
    ----------
    filters : Theano shared variable
        4-tensor of shape (in channels, rows, cols, out channels)
    input_axes : WRITEME
    batch_size : WRITEME
    output_axes : WRITEME
    kernel_stride : WRITEME
    pad : WRITEME
    message : WRITEME
    partial_sum : WRITEME
    """

    def __init__(self, filters, input_axes=('c', 0, 1, 'b'),
                 batch_size=None, output_axes=('c', 0, 1, 'b'),
                 kernel_stride=(1, 1), pad=0, message='',
                 partial_sum=None):
        if len(kernel_stride) != 2:
            raise ValueError("kernel_stride must have length 2")
        elif kernel_stride[0] != kernel_stride[1]:
            raise ValueError("only values of kernel_stride with both "
                             "elements equal are supported currently")
        if message != '':
            raise NotImplementedError()

        if batch_size is not None:
            raise NotImplementedError()

        self.input_axes = input_axes
        self.output_axes = output_axes

        # filters should be a GPU shared variable.
        # I guess you could GpuFromHost them every time,
        # but if you're using this class you probably care
        # about performance and want to be at least warned
        # that this is happening
        assert hasattr(filters, 'get_value')
        assert 'Cuda' in str(type(filters))
        self._filters = filters
        self.pad = pad
        self.partial_sum = partial_sum
        self.kernel_stride = kernel_stride

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._filters]

    @functools.wraps(LinearTransform.get_weights_topo)
    def get_weights_topo(self, borrow=False):
        """
        .. todo::

            WRITEME
        """
        inp, rows, cols, outp = range(4)
        raw = self._filters.get_value(borrow=borrow)
        return np.transpose(raw, (outp, rows, cols, inp))

    def lmul(self, x):
        """
        .. todo::

            WRITEME properly

        dot(x, A)
        aka, do convolution with input image x
        """

        check_cuda(str(type(self)) + ".lmul")

        cpu = 'Cuda' not in str(type(x))

        if cpu:
            x = gpu_from_host(x)

        # x must be formatted as channel, topo dim 0, topo dim 1, batch_index
        # for use with FilterActs
        assert x.ndim == 4
        x_axes = self.input_axes
        assert len(x_axes) == 4

        op_axes = ('c', 0, 1, 'b')

        if tuple(x_axes) != op_axes:
            x = x.dimshuffle(*[x_axes.index(axis) for axis in x_axes])

        x = gpu_contiguous(x)

        # Patch old pickle files.
        if not hasattr(self, 'kernel_stride'):
            self.kernel_stride = (1, 1)
        rval = FilterActs(self.pad, self.partial_sum, self.kernel_stride[0])(
            x,
            self._filters
        )

        # Format the output based on the output space
        rval_axes = self.output_axes
        assert len(rval_axes) == 4

        if cpu:
            rval = host_from_gpu(rval)

        if tuple(rval_axes) != op_axes:
            rval = rval.dimshuffle(*[op_axes.index(axis)
                                     for axis in rval_axes])

        return rval

    def lmul_T(self, x):
        """
        .. todo::

            WRITEME
        """

        check_cuda(str(type(self)) + ".lmul_T")

        assert x.dtype == self._filters.dtype

        op_axes = ('c', 0, 1, 'b')
        axes = self.output_axes
        if tuple(axes) != op_axes:
            x = x.dimshuffle(*[axes.index(ax) for ax in op_axes])

        x = gpu_contiguous(x)

        rval = ImageActs(pad=self.pad, partial_sum=self.partial_sum,
                         stride=self.kernel_stride[0])(x, self._filters)

        # Format the output based on the input space
        axes = self.input_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(op_axes.index(axes[0]),
                                   op_axes.index(axes[1]),
                                   op_axes.index(axes[2]),
                                   op_axes.index(axes[3]))

        return rval

    def lmul_sq_T(self, x):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError("This method is not yet modified since "
                                  "copy-pasting from pylearn2.linear.conv2d")
        """ Kind of a stupid hacky method used to support convolutional score
        matching. Ought to find a way to make _filters symbolic rather than
        shared.
        """
        assert x.dtype == self._filters.dtype

        op_axes = ('b', 'c', 0, 1)
        axes = self.output_axes
        if tuple(axes) != op_axes:
            x = x.dimshuffle(axes.index('b'), axes.index('c'),
                             axes.index(0), axes.index(1))

        # dot(x, sq(A).T)
        dummy_v = T.tensor4()
        sqfilt = T.square(self._filters)
        z_hs = 0.  # conv2d(dummy_v, sqfilt,
                   # image_shape=self._img_shape,
                   # filter_shape=self._filters_shape,
                   # kernel_stride=self._kernel_stride,
                   # pad = self.pad
                   # )
        rval, xdummy = z_hs.owner.op.grad((dummy_v, sqfilt), (x,))

        # Format the output based on the input space
        axes = self.input_space.axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(op_axes.index(axes[0]),
                                   op_axes.index(axes[1]),
                                   op_axes.index(axes[2]),
                                   op_axes.index(axes[3]))

        return rval

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        pass


def make_random_conv2D(irange, input_channels, input_axes, output_axes,
                       output_channels, kernel_shape, kernel_stride=(1, 1),
                       pad=0, message="", rng=None, partial_sum=None,
                       sparse_init=None):
    """
    .. todo::

        WRITEME properly

    Creates a Conv2D with random kernels.
    Should be functionally equivalent to
    pylearn2.linear.conv2d.make_random_conv2D
    """

    rng = make_np_rng(rng, default_seed, which_method='uniform')

    W = sharedX(rng.uniform(-irange, irange,
                            (input_channels, kernel_shape[0],
                             kernel_shape[1], output_channels)))

    return Conv2D(filters=W, input_axes=input_axes, output_axes=output_axes,
                  kernel_stride=kernel_stride, pad=pad, message=message,
                  partial_sum=partial_sum)


def make_sparse_random_conv2D(num_nonzero, input_space, output_space,
                              kernel_shape, pad=0, kernel_stride=(1, 1),
                              border_mode='valid', message="", rng=None,
                              partial_sum=None):
    """
    .. todo::

        WRITEME properly

    Creates a Conv2D with random kernels, where the randomly initialized
    values are sparse
    """

    rng = make_np_rng(rng, default_sparse_seed,
                      which_method=['randn', 'randint'])

    W = np.zeros((input_space.num_channels, kernel_shape[0],
                  kernel_shape[1], output_space.num_channels))

    def random_coord():
        return [rng.randint(dim) for dim in W.shape[0:3]]

    for o in xrange(output_space.num_channels):
        for i in xrange(num_nonzero):
            ch, r, c = random_coord()
            while W[ch, r, c, o] != 0:
                ch, r, c = random_coord()
            W[ch, r, c, o] = rng.randn()

    W = sharedX(W)

    return Conv2D(filters=W, input_axes=input_space.axes,
                  output_axes=output_space.axes, kernel_stride=kernel_stride,
                  pad=pad, message=message, partial_sum=partial_sum)


def setup_detector_layer_c01b(layer, input_space, rng, irange="not specified"):
    """
    .. todo::

        WRITEME properly

    Takes steps to set up an object for use as being some kind of convolutional
    layer. This function sets up only the detector layer.

    Does the following:

    * raises a RuntimeError if cuda is not available
    * sets layer.input_space to input_space
    * sets up addition of dummy channels for compatibility with cuda-convnet:

      - layer.dummy_channels: # of dummy channels that need to be added
        (You might want to check this and raise an Exception if it's not 0)
      - layer.dummy_space: The Conv2DSpace representing the input with dummy
        channels added

    * sets layer.detector_space to the space for the detector layer
    * sets layer.transformer to be a Conv2D instance
    * sets layer.b to the right value

    Parameters
    ----------
    layer : object
        Any python object that allows the modifications described below and
        has the following attributes:

          * pad : int describing amount of zero padding to add
          * kernel_shape : 2-element tuple or list describing spatial shape of
            kernel
          * fix_kernel_shape : bool, if true, will shrink the kernel shape to
            make it feasible, as needed (useful for hyperparameter searchers)
          * detector_channels : The number of channels in the detector layer
          * init_bias : numeric constant added to a tensor of zeros to
            initialize the bias
          * tied_b : If true, biases are shared across all spatial locations
    input_space : WRITEME
        A Conv2DSpace to be used as input to the layer
    rng : WRITEME
        A numpy RandomState or equivalent
    """

    if irange != "not specified":
        raise AssertionError(
            "There was a bug in setup_detector_layer_c01b."
            "It uses layer.irange instead of the irange parameter to the "
            "function. The irange parameter is now disabled by this "
            "AssertionError, so that this error message can alert you that "
            "the bug affected your code and explain why the interface is "
            "changing. The irange parameter to the function and this "
            "error message may be removed after April 21, 2014."
        )

    # Use "self" to refer to layer from now on, so we can pretend we're
    # just running in the set_input_space method of the layer
    self = layer

    # Make sure cuda is available
    check_cuda(str(type(self)))

    # Validate input
    if not isinstance(input_space, Conv2DSpace):
        raise TypeError("The input to a convolutional layer should be a "
                        "Conv2DSpace, but layer " + self.layer_name + " got " +
                        str(type(self.input_space)))

    if not hasattr(self, 'detector_channels'):
        raise ValueError("layer argument must have a 'detector_channels' "
                         "attribute specifying how many channels to put in "
                         "the convolution kernel stack.")

    # Store the input space
    self.input_space = input_space

    # Make sure number of channels is supported by cuda-convnet
    # (multiple of 4 or <= 3)
    # If not supported, pad the input with dummy channels
    ch = self.input_space.num_channels
    rem = ch % 4
    if ch > 3 and rem != 0:
        self.dummy_channels = 4 - rem
    else:
        self.dummy_channels = 0
    self.dummy_space = Conv2DSpace(
        shape=input_space.shape,
        channels=input_space.num_channels + self.dummy_channels,
        axes=('c', 0, 1, 'b')
    )

    if hasattr(self, 'kernel_stride'):
        kernel_stride = self.kernel_stride
    else:
        kernel_stride = [1, 1]

    output_shape = \
        [int(np.ceil((i_sh + 2. * self.pad - k_sh) / float(k_st))) + 1
         for i_sh, k_sh, k_st in zip(self.input_space.shape,
                                     self.kernel_shape, kernel_stride)]

    def handle_kernel_shape(idx):
        if self.kernel_shape[idx] < 1:
            raise ValueError("kernel must have strictly positive size on all "
                             "axes but has shape: " + str(self.kernel_shape))
        if output_shape[idx] <= 0:
            if self.fix_kernel_shape:
                self.kernel_shape[idx] = \
                    self.input_space.shape[idx] + 2 * self.pad
                assert self.kernel_shape[idx] != 0
                output_shape[idx] = 1
                warnings.warn("Had to change the kernel shape to make "
                              "network feasible")
            else:
                raise ValueError("kernel too big for input "
                                 "(even with zero padding)")

    map(handle_kernel_shape, [0, 1])

    if self.detector_channels < 16:
        raise ValueError("Cuda-convnet requires the detector layer to have "
                         "at least 16 channels.")

    self.detector_space = Conv2DSpace(shape=output_shape,
                                      num_channels=self.detector_channels,
                                      axes=('c', 0, 1, 'b'))

    if hasattr(self, 'partial_sum'):
        partial_sum = self.partial_sum
    else:
        partial_sum = 1

    if hasattr(self, 'sparse_init') and self.sparse_init is not None:
        self.transformer = \
            checked_call(make_sparse_random_conv2D,
                         OrderedDict([('num_nonzero', self.sparse_init),
                                      ('input_space', self.input_space),
                                      ('output_space', self.detector_space),
                                      ('kernel_shape', self.kernel_shape),
                                      ('pad', self.pad),
                                      ('partial_sum', partial_sum),
                                      ('kernel_stride', kernel_stride),
                                      ('rng', rng)]))
    else:
        self.transformer = make_random_conv2D(
            irange=self.irange,
            input_axes=self.input_space.axes,
            output_axes=self.detector_space.axes,
            input_channels=self.dummy_space.num_channels,
            output_channels=self.detector_space.num_channels,
            kernel_shape=self.kernel_shape,
            pad=self.pad,
            partial_sum=partial_sum,
            kernel_stride=kernel_stride,
            rng=rng
        )

    W, = self.transformer.get_params()
    W.name = self.layer_name + '_W'

    if self.tied_b:
        self.b = sharedX(np.zeros(self.detector_space.num_channels) +
                         self.init_bias)
    else:
        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
    self.b.name = self.layer_name + '_b'

    logger.info('Input shape: {0}'.format(self.input_space.shape))
    logger.info('Detector space: {0}'.format(self.detector_space.shape))
