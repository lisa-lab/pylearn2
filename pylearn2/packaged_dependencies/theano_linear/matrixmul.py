import numpy
import theano
from theano import tensor

from .linear import LinearTransform
from .util import ndarray_status
from .util import tile_slices_to_image

class MatrixMul(LinearTransform):
    """
    Linear transform backed by an actual matrix.

    row_shape and col_shape are be tuples of any length

    Matrix shared variable self._W must have a number of rows that is the
    product of row_shape and a number of columns that is the product of
    col_shape.

    W can be a sparse matrix too.
    """

    # Works for Sparse and TensorType matrices
    def __init__(self, W, row_shape=None, col_shape=None, params=None):
        """

        If W is not shared variable, row_shape and col_shape must be
        specified.
        """
        if params is None:
            params = [W]
        super(MatrixMul, self).__init__(params)
        self._W = W
        Wval = None
        if row_shape is None or col_shape is None:
            Wval = W.get_value(borrow=True)
            rows, cols = Wval.shape
            if row_shape is None:
                self.__row_shape = cols,
            else:
                self.__row_shape = tuple(row_shape)
                if numpy.prod(self.__row_shape) != cols:
                    raise ValueError('invalid row_shape: prod != %i' % cols,
                            self.__row_shape)
            if col_shape is None:
                self.__col_shape = rows,
            else:
                self.__col_shape = tuple(col_shape)
                if numpy.prod(self.__col_shape) != rows:
                    raise ValueError('invalid col_shape: prod != %i' % rows,
                            self.__col_shape)

    def lmul(self, x):
        # dot(x, A)
        RR, CC = self.split_left_shape(tuple(x.shape), T=False)
        xW = theano.dot(
                x.reshape((tensor.mul(*RR), tensor.mul(*CC))),
                self._W)
        rshape = self.row_shape()
        yshp = tensor.stack(*(RR + rshape))
        rval = xW.reshape(yshp, ndim=len(RR) + len(rshape))
        return rval

    def lmul_T(self, x):
        CC, RR = self.split_right_shape(tuple(x.shape), T=True)
        x_WT = theano.dot(
                x.reshape((tensor.mul(*CC), tensor.mul(*RR))),
                self._W.T)
        cshape = self.col_shape()
        yshp = tensor.stack(*(CC + cshape))
        rval = x_WT.reshape(yshp, ndim=len(CC) + len(cshape))
        return rval

    def row_shape(self):
        return self.__row_shape

    def col_shape(self):
        return self.__col_shape

    def print_status(self):
        print ndarray_status(self._W.get_value(borrow=True), msg=self._W.name)

    def tile_columns(self, channel_major=False, scale_each=False,
            min_dynamic_range=1e-4, **kwargs):
        W = self._W.get_value(borrow=False).T
        shape = self.row_shape()
        if channel_major:
            W.shape = (W.shape[0:1]+shape)
            W = W.transpose(0,2,3,1) #put colour last
        else:
            raise NotImplementedError()

        return tile_slices_to_image(W,
                scale_each=scale_each,
                **kwargs)

