from pylearn2.packaged_dependencies.theano_linear.conv2d import Conv2d as OrigConv2D
import theano.tensor as T
from pylearn2.utils import sharedX
import numpy as np
from theano.tensor.nnet.conv import conv2d
from pylearn2.linear.linear_transform import LinearTransform as P2LT
import functools

class Conv2D(OrigConv2D):
    """ Extend the TheanoLinear Conv2d class to support everything
    needed for a pylearn2 linear operator.

    Also extend it to handle different axis semantics."""

    def __init__(self,
            filters,
            batch_size,
            input_space,
            output_axes = ('b',0,1,'c'),
        subsample = (1, 1), border_mode = 'valid',
        filters_shape = None, message = ''):

        self.input_space = input_space
        self.output_axes = output_axes

        super(Conv2D,self).__init__(filters = filters,
                img_shape = (batch_size, input_space.nchannels,\
                    input_space.shape[0], input_space.shape[1]),
                subsample = subsample,
                border_mode = border_mode,
                filters_shape = filters.get_value(borrow=True).shape,
                message = message)

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        return set([ self._filters ])

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self,borrow):
        return np.transpose(self._filters.get_value(borrow = borrow),(0,2,3,1))

    def lmul(self, x):
        """
        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders """

        # x must be formatted as batch index, channel, topo dim 0, topo dim 1
        # for use with conv2d
        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1))


        rval =  conv2d(
                x, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )

        # Format the output based on the output space
        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                    op_axes.index(axes[0]),
                    op_axes.index(axes[1]),
                    op_axes.index(axes[2]),
                    op_axes.index(axes[3]))

        return rval

    def lmul_T(self, x):
        """ override the original Conv2D lmul_T to make it work
        with pylearn format of topological data using dimshuffles """
        assert x.dtype == self._filters.dtype

        op_axes = ('b', 'c', 0, 1)
        axes = self.output_axes
        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                    axes.index('b'),
                    axes.index('c'),
                    axes.index(0),
                    axes.index(1))

        # dot(x, A.T)
        dummy_v = T.tensor4()
        z_hs = conv2d(dummy_v, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )
        rval, xdummy = z_hs.owner.op.grad((dummy_v, self._filters), (x,))


        # Format the output based on the input space
        axes = self.input_space.axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                    op_axes.index(axes[0]),
                    op_axes.index(axes[1]),
                    op_axes.index(axes[2]),
                    op_axes.index(axes[3]))

        return rval

    def lmul_sq_T(self, x):
        """ Kind of a stupid hacky method used to support convolutional score matching.
        Ought to find a way to make _filters symbolic rather than shared.
        """
        assert x.dtype == self._filters.dtype

        op_axes = ('b', 'c', 0, 1)
        axes = self.output_axes
        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                    axes.index('b'),
                    axes.index('c'),
                    axes.index(0),
                    axes.index(1))

        # dot(x, sq(A).T)
        dummy_v = T.tensor4()
        sqfilt = T.square(self._filters)
        z_hs = conv2d(dummy_v, sqfilt,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )
        rval, xdummy = z_hs.owner.op.grad((dummy_v, sqfilt), (x,))

        # Format the output based on the input space
        axes = self.input_space.axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                    op_axes.index(axes[0]),
                    op_axes.index(axes[1]),
                    op_axes.index(axes[2]),
                    op_axes.index(axes[3]))

        return rval

    def set_batch_size(self, batch_size):
        self._img_shape = tuple([ batch_size ] + list(self._img_shape[1:]))



def make_random_conv2D(irange, input_space, output_space,
        kernel_shape, batch_size, \
        subsample = (1,1), border_mode = 'valid', message = ""):
    """ Creates a Conv2D with random kernels """


    rng = np.random.RandomState([1,2,3])

    W = sharedX( rng.uniform(-irange,irange,( output_space.nchannels, input_space.nchannels, \
            kernel_shape[0], kernel_shape[1])))

    return Conv2D(filters = W,
        batch_size = batch_size,
        input_space = input_space,
        output_axes = output_space.axes,
        subsample = subsample, border_mode = border_mode,
        filters_shape = W.get_value(borrow=True).shape, message = message)


