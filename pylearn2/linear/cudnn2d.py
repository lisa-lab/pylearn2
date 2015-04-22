"""
A module for convolutions with cudnn.
"""

__author__ = "Nicolas Ballas"
__license__ = "3-clause BSD"
__credits__ = "Nicolas Ballas and Francesco Visin"
__maintainer__ = "Lisa Lab"

import functools
import numpy as np

from theano.sandbox.cuda.dnn import GpuDnnConv, GpuDnnConvDesc
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty

from pylearn2.packaged_dependencies.theano_linear.conv2d \
    import Conv2d as OrigConv2D

from pylearn2.linear.linear_transform import LinearTransform as P2LT
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


default_seed = [2012, 11, 6, 9]
default_sparse_seed = [2012, 11, 6]


class Cudnn2D(OrigConv2D):
    """
    Wrapper on the Theano Cudnn op.

    Parameters
    ----------
    filters : Theano shared variable
        4D-tensor of shape (out channels, in channels, rows, cols)
    batch_size : int
        The size of the input batches
    input_space : Space
        The Space of the input data
    output_axes : tuple, optional
        The requested output axes. If not specified `bc01` will be used.
    subsample : tuple or list, optional
        Factor by which to subsample the output. Default (1, 1)
    border_mode : string, optional
        `valid` or `full`. See scipy.signal.convolve2d
    filters_shape : tuple of length 2 or 3, optional
        ([filter's number,] filter's height, filter's width)
    message : string, optional
        TODO
    """

    def __init__(self,
                 filters,
                 batch_size,
                 input_space,
                 output_axes=('b', 'c', 0, 1),
                 subsample=(1, 1),
                 border_mode='valid',
                 filters_shape=None,
                 message=''):

        assert batch_size is None or batch_size > 0
        self._input_space = input_space
        self._output_axes = output_axes
        self._subsample = tuple(subsample)
        self._border_mode = border_mode

        super(Cudnn2D, self).__init__(
            filters=filters,
            img_shape=(batch_size, input_space.num_channels,
                       input_space.shape[0], input_space.shape[1]),
            subsample=self._subsample,
            border_mode=border_mode,
            filters_shape=filters.get_value(borrow=True).shape,
            message=message
        )

        # conv_op has to be changed
        self._conv_op = GpuDnnConv()
        self._desc = GpuDnnConvDesc(border_mode=border_mode,
                                    subsample=self._subsample,
                                    conv_mode='conv')

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """ Return self._filters. """
        return [self._filters]

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self, borrow):
        """
        Parameters
        ----------
        borrow : TODO
            TODO
        """
        return np.transpose(self._filters.get_value(borrow=borrow),
                            (0, 2, 3, 1))

    def lmul(self, x):
        """
        .. todo::

            WRITEME properly

        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders

        Parameters
        ----------
        x : TODO
            TODO
        """
        # x must be formatted as batch index, channel, topo dim 0, topo dim 1
        # for use with conv2d, so check what the current input space format is
        assert x.ndim == 4
        axes = self._input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(*[axes.index(ax) for ax in op_axes])

        # The calling format has to be changed
        img = gpu_contiguous(x)
        kerns = gpu_contiguous(self._filters)
        shape = GpuDnnConv.get_out_shape(
            img.shape, kerns.shape, self._border_mode, self._subsample)
        rval = gpu_alloc_empty(*shape)
        desc = self._desc(img.shape, kerns.shape)
        rval = self._conv_op(img, kerns, rval, desc)

        # Format the output based on the output space
        axes = self._output_axes
        assert len(axes) == 4

        if tuple(self._output_axes) != op_axes:
            rval = rval.dimshuffle(*[op_axes.index(ax) for ax in
                                     self._output_axes])

        return rval

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : TODO
            TODO
        """
        self._img_shape = tuple([batch_size] + list(self._img_shape[1:]))


def make_random_conv2D(irange,
                       input_space,
                       output_space,
                       kernel_shape,
                       batch_size=None,
                       subsample=(1, 1),
                       border_mode = 'valid',
                       message = "",
                       rng = None):
    """
    .. todo::

        WRITEME properly

    Creates a CorrMM2D with random kernels

    Parameters
    ----------
    irange : TODO
    input_space : TODO
    output_space : TODO
    kernel_shape : 2D list or tuple
    batch_size : int, optional
    subsample : tuple, optional
    border_mode : string, optional
    message : string, optional
    rng : optional
    """

    rng = make_np_rng(rng, default_seed, which_method='uniform')

    W = sharedX(rng.uniform(-irange, irange, (output_space.num_channels,
                                              input_space.num_channels,
                                              kernel_shape[0],
                                              kernel_shape[1])))

    return Cudnn2D(
        filters=W,
        batch_size=batch_size,
        input_space=input_space,
        output_axes=output_space.axes,
        subsample=tuple(subsample),
        border_mode=border_mode,
        filters_shape=W.get_value(borrow=True).shape,
        message=message
    )


def make_sparse_random_conv2D(num_nonzero,
                              input_space,
                              output_space,
                              kernel_shape,
                              batch_size,
                              subsample=(1, 1),
                              border_mode='valid',
                              message="",
                              rng=None):

    """
    .. todo::

        WRITEME properly

    Creates a Cudnn2D with random kernels, where the randomly initialized
    values are sparse

    Parameters
    ----------
    num_nonzero : TODO
    input_space : TODO
    output_space : TODO
    kernel_shape : TODO
    batch_size : TODO
    subsample : TODO, optional
    border_mode : TODO, optional
    message : TODO, optional
    rng : TODO, optional

    """

    raise AssertionError(
        "TODO: I think this is a bug--num_nonzero "
        "determines the total number of nonzero elements in the "
        "whole kernel stack, not the number of non-zero elements per "
        "kernel. Investigate what it's meant to do."
    )

    rng = make_np_rng(rng, default_sparse_seed,
                      which_method=['randn', 'randint'])

    W = np.zeros((output_space.num_channels, input_space.num_channels,
                  kernel_shape[0], kernel_shape[1]))

    def random_coord():
        return [rng.randint(dim) for dim in W.shape]

    for i in range(num_nonzero):
        o, ch, r, c = random_coord()
        while W[o, ch, r, c] != 0:
            o, ch, r, c = random_coord()
        W[o, ch, r, c] = rng.randn()

    W = sharedX(W)

    return Cudnn2D(
        filters=W,
        batch_size=batch_size,
        input_space=input_space,
        output_axes=output_space.axes,
        subsample=subsample,
        border_mode=border_mode,
        filters_shape=W.get_value(borrow=True).shape,
        message=message
    )
