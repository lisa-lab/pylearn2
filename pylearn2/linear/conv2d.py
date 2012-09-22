from pylearn2.packaged_dependencies.theano_linear.conv2d import Conv2d as OrigConv2D
import theano.tensor as T
from pylearn2.utils import sharedX
import numpy as np
from theano.tensor.nnet.conv import conv2d
from pylearn2.linear.linear_transform import LinearTransform as P2LT
import functools

class Conv2D(OrigConv2D):
    """ Extend the TheanoLinear Conv2d class to support everything
    needed for a pylearn2 linear operator """

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        return set([ self._filters ])

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self,borrow):
        return np.transpose(self._filters.get_value(borrow = borrow),(0,2,3,1))

    def lmul(self, x):
        """ override the original Conv2D lmul to make it work
        with pylearn format of topological data using dimshuffles """
        # dot(x, A)

        x = x.dimshuffle(0,3,1,2)

        rval =  conv2d(
                x, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )

        return rval.dimshuffle(0,2,3,1)

    def lmul_T(self, x):
        """ override the original Conv2D lmul_T to make it work
        with pylearn format of topological data using dimshuffles """
        assert x.dtype == self._filters.dtype

        x = x.dimshuffle(0,3,1,2)

        # dot(x, A.T)
        dummy_v = T.tensor4()
        z_hs = conv2d(dummy_v, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )
        xfilters, xdummy = z_hs.owner.op.grad((dummy_v, self._filters), (x,))

        xfilters = xfilters.dimshuffle(0,2,3,1)

        return xfilters

    def lmul_sq_T(self, x):
        """ Kind of a stupid hacky method used to support convolutional score matching.
        Ought to find a way to make _filters symbolic rather than shared.
        """
        assert x.dtype == self._filters.dtype

        x = x.dimshuffle(0,3,1,2)

        # dot(x, sq(A).T)
        dummy_v = T.tensor4()
        sqfilt = T.square(self._filters)
        z_hs = conv2d(dummy_v, sqfilt,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )
        xfilters, xdummy = z_hs.owner.op.grad((dummy_v, sqfilt), (x,))

        xfilters = xfilters.dimshuffle(0,2,3,1)

        return xfilters

    def set_batch_size(self, batch_size):
        self._img_shape = tuple([ batch_size ] + list(self._img_shape[1:]))



def make_random_conv2D(irange, input_space, output_space,
        kernel_shape, batch_size, \
        subsample = (1,1), border_mode = 'valid', message = ""):
    """ Creates a Conv2D with random kernels """


    rng = np.random.RandomState([1,2,3])

    W = sharedX( rng.uniform(-irange,irange,( output_space.nchannels, input_space.nchannels, \
            kernel_shape[0], kernel_shape[1])))

    return Conv2D(filters = W, \
        img_shape = (batch_size, input_space.nchannels,\
                    input_space.shape[0], input_space.shape[1]), \
        subsample = subsample, border_mode = border_mode, \
        filters_shape = W.get_value(borrow=True).shape, message = message)


