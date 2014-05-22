"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from theano import tensor as T

from pylearn2.linear.linear_transform import LinearTransform
import functools
import numpy as np
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


class MatrixMul(LinearTransform):
    """
    The most basic LinearTransform: matrix multiplication. See TheanoLinear
    for more documentation.

    Note: this does not inherit from the TheanoLinear's MatrixMul.

    The TheanoLinear version does a bunch of extra undocumented reshaping,
    concatenating, etc. that looks like it's probably meant to allow converting
    between Spaces without warning. Since the reshape and concatenate
    operations
    are always inserted whether they're needed or not, this can cause annoying
    things like the reshape breaking if you change the shape of W, bugs in
    Theano's optimization system being harder to avoid, etc.

    Parameters
    ----------
    W : WRITEME
    """

    def __init__(self, W):
        """
        Sets the initial values of the matrix
        """
        self._W = W

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._W]

    def lmul(self, x):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        x : ndarray, 1d or 2d
            The input data
        """

        return T.dot(x, self._W)

    def lmul_T(self, x):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        x : ndarray, 1d or 2d
            The input data
        """
        return T.dot(x, self._W.T)


def make_local_rfs(dataset, nhid, rf_shape, stride, irange = .05,
        draw_patches = False, rng = None):
    """
    Initializes a weight matrix with local receptive fields

    Parameters
    ----------
    dataset : pylearn2.datasets.dataset.Dataset
        Dataset defining the topology of the space (needed to convert 2D
        patches into subsets of pixels in a 1D filter vector)
    nhid : int
        Number of hidden units to make filters for
    rf_shape : list or tuple (2 elements)
        Gives topological shape of a receptive field
    stride : list or tuple (2 elements)
        Gives offset between receptive fields
    irange : float
        If draw_patches is False, weights are initialized in U(-irange,irange)
    draw_patches : bool
        If True, weights are drawn from random examples

    Returns
    -------
    weights : ndarray
        2D ndarray containing the desired weights.
    """
    s = dataset.view_shape()
    height, width, channels = s
    W_img = np.zeros( (nhid, height, width, channels) )

    last_row = s[0] - rf_shape[0]
    last_col = s[1] - rf_shape[1]

    rng = make_np_rng(rng, [2012,07,18], which_method='uniform')


    if stride is not None:
        # local_rf_stride specified, make local_rfs on a grid
        assert last_row % stride[0] == 0
        num_row_steps = last_row / stride[0] + 1

        assert last_col % stride[1] == 0
        num_col_steps = last_col /stride[1] + 1

        total_rfs = num_row_steps * num_col_steps

        if nhid % total_rfs != 0:
            raise ValueError('nhid modulo total_rfs should be 0, but we get '
                    '%d modulo %d = %d' % (nhid, total_rfs, nhid % total_rfs))

        filters_per_rf = nhid / total_rfs

        idx = 0
        for r in xrange(num_row_steps):
            rc = r * stride[0]
            for c in xrange(num_col_steps):
                cc = c * stride[1]

                for i in xrange(filters_per_rf):

                    if draw_patches:
                        img = dataset.get_batch_topo(1)[0]
                        local_rf = img[rc:rc+rf_shape[0],
                                       cc:cc+rf_shape[1],
                                       :]
                    else:
                        local_rf = rng.uniform(-irange,
                                    irange,
                                    (rf_shape[0], rf_shape[1], s[2]) )



                    W_img[idx,rc:rc+rf_shape[0],
                      cc:cc+rf_shape[1],:] = local_rf
                    idx += 1
        assert idx == nhid
    else:
        raise NotImplementedError()
        #the case below is copy-pasted from s3c and not generalized yet
        #no stride specified, use random shaped patches
        """
        assert local_rf_max_shape is not None

        for idx in xrange(nhid):
            shape = [ self.rng.randint(min_shape,max_shape+1) for
                    min_shape, max_shape in zip(
                        local_rf_shape,
                        local_rf_max_shape) ]
            loc = [ self.rng.randint(0, bound - width + 1) for
                    bound, width in zip(s, shape) ]

            rc, cc = loc

            if local_rf_draw_patches:
                img = local_rf_src.get_batch_topo(1)[0]
                local_rf = img[rc:rc+shape[0],
                               cc:cc+shape[1],
                               :]
            else:
                local_rf = self.rng.uniform(-self.irange,
                            self.irange,
                            (shape[0], shape[1], s[2]) )

            W_img[idx,rc:rc+shape[0],
                      cc:cc+shape[1],:] = local_rf
        """


    W = dataset.view_converter.topo_view_to_design_mat(W_img).T

    rval = MatrixMul(W = sharedX(W))

    return rval
