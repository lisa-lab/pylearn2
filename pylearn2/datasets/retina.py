"""
Retina-inspired preprocessing as described in
    Salakhutdinov, R. and Hinton, G. Deep Boltzmann machines.
    In *AISTATS* 2009.
"""
import numpy
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.space import Conv2DSpace


def foveate_channel(img, rings, output, start_idx):
    """For a given channel (image), perform pooling on peripheral vision.

    .. todo::

        Write parameter list
    """
    ring_w = numpy.sum(rings)

    # extract image center, which remains dense
    inner_img = img[:, ring_w : img.shape[1] - ring_w , ring_w : img.shape[2] - ring_w]

    # flatten and write to dense output matrix
    inner_img = inner_img.reshape(len(output), -1)
    end_idx = start_idx + inner_img.shape[1]
    output[:, start_idx : end_idx] = inner_img

    # start by downsampling the periphery of the images
    idx = 0
    start_idx = end_idx
    for rd in rings:
        # downsample the ring with top-left corner (idx,idx) of width rd
        # results are written in output[start_idx:]
        start_idx = downsample_ring(img, idx, rd, output, start_idx)
        idx += rd

    return start_idx

def downsample_ring(img, coord, width, output, start_idx):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    img : WRITEME
        numpy matrix in topological order
        (batch size, rows, cols, channels)
    coord : WRITEME
        perform average pooling starting at coordinate (coord,coord)
    width : WRITEME
        width of "square ring" to average pool
    output : WRITEME
        dense design matrix, of shape (batch size, rows*cols*channels)
    start_idx : WRITEME
        column index where to start writing the output
    """
    (img_h,img_w) = img.shape[1:3]

    # left column, full height
    start_idx = downsample_rect(img, coord, coord, img_h - coord, coord + width, width, output, start_idx)
    # right column, full height
    start_idx = downsample_rect(img, coord, img_w - coord - width, img_h - coord, img_w - coord, width, output, start_idx)
    # top row, between columns
    start_idx = downsample_rect(img, coord, coord + width, coord + width, img_w - coord - width, width, output, start_idx)
    # bottom row, between columns
    start_idx = downsample_rect(img, img_h - coord - width, coord + width, img_h - coord, img_w - coord - width, width, output, start_idx)

    return start_idx


def downsample_rect(img, start_row, start_col, end_row, end_col, width, output, start_idx):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    img : WRITEME
        numpy matrix in topological order
        (batch size, rows, cols, channels)
    start_row : WRITEME
        row index of top-left corner of rectangle to average pool
    start_col : WRITEME
        col index of top-left corner of rectangle to average pool
    end_row : WRITEME
        row index of bottom-right corner of rectangle to average pool
    end_col : WRITEME
        col index of bottom-right corner of rectangle to average pool
    width : WRITEME
        take the mean over rectangular block of this width
    output : WRITEME
        dense design matrix, of shape (batch size, rows*cols*channels)
    start_idx : WRITEME
        column index where to start writing the output
    """
    idx = start_idx

    for i in xrange(start_row, end_row - width + 1, width):
        for j in xrange(start_col, end_col - width + 1, width):
            block = img[:, i:i+width, j:j+width]
            output[:,idx] = numpy.apply_over_axes(numpy.mean, block, axes=[1,2])[:,0,0]
            idx += 1

    return idx


def defoveate_channel(img, rings, dense_input, start_idx):
    """
    Defoveate a single channel of the DenseDesignMatrix dense_input into the
    variable, stored in topological ordering.

    Parameters
    ----------
    img : WRITEME
        channel for defoveated image of shape (batch, img_h, img_w)
    rings : WRITEME
        list of ring_sizes which were used to generate dense_input
    dense_input : WRITEME
        DenseDesignMatrix containing foveated dataset, of shape 
        (batch, dims)
    start_idx : WRITEME
        channel pointed to by img starts at dense_input[start_idx]
    """
    ring_w = numpy.sum(rings)

    # extract image center, which remains dense
    inner_h = img.shape[1] - 2*ring_w
    inner_w = img.shape[2] - 2*ring_w
    end_idx = start_idx + inner_h * inner_w
    inner_img = dense_input[:, start_idx : end_idx].reshape(-1, inner_h, inner_w)

    # now restore image center in uncompressed image
    img[:, ring_w:ring_w+inner_h, ring_w:ring_w+inner_w] = inner_img

    # now undo downsampling along the periphery
    idx = 0
    start_idx = end_idx
    for rd in rings:
        # downsample the ring with top-left corner (idx,idx) of width rd
        # results are written in img[idx:idx+rd, idx:idx+rd]
        start_idx = restore_ring(img, idx, rd, dense_input, start_idx)
        idx += rd

    return start_idx


def restore_ring(output, coord, width, dense_input, start_idx):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    output : WRITEME
        output matrix in topological order
        (batch, height, width, channels)
    coord : WRITEME
        perform average pooling starting at coordinate (coord,coord)
    width : WRITEME
        width of "square ring" to average pool
    dense_input : WRITEME
        dense design matrix to convert (batchsize, dims)
    start_idx : WRITEME
        column index where to start writing the output
    """
    (img_h, img_w) = output.shape[1:3]

    # left column, full height
    start_idx = restore_rect(output, coord, coord, img_h - coord, coord + width, width, dense_input, start_idx)
    # right column, full height
    start_idx = restore_rect(output, coord, img_w - coord - width, img_h - coord, img_w - coord, width, dense_input, start_idx)
    # top row, between columns
    start_idx = restore_rect(output, coord, coord + width, coord + width, 96 - coord - width, width, dense_input, start_idx)
    # bottom row, between columns
    start_idx = restore_rect(output, img_h - coord - width, coord + width, img_h - coord, img_w - coord - width, width, dense_input, start_idx)

    return start_idx


def restore_rect(output, start_row, start_col, stop_row, stop_col, width, dense_input, start_idx):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    output : WRITEME
        output matrix in topological order
        (batch, height, width, channels)
    start_row : WRITEME
        row index of top-left corner of rectangle to average pool
    start_col : WRITEME
        col index of top-left corner of rectangle to average pool
    end_row : WRITEME
        row index of bottom-right corner of rectangle to average pool
    end_col : WRITEME
        col index of bottom-right corner of rectangle to average pool
    width : WRITEME
        take the mean over rectangular block of this width
    dense_input : WRITEME
        dense design matrix to convert (batchsize, dims)
    start_idx : WRITEME
        column index where to start writing the output
    """
    idx = start_idx

    for i in xrange(start_row, stop_row - width + 1, width):
        for j in xrange(start_col, stop_col - width + 1, width):
            # broadcast along the width and height of the block
            output[:, i:i+width, j:j+width] = dense_input[:, idx][:, None, None]
            idx += 1

    return idx


def get_encoded_size(img_h, img_w, rings):
    """
    .. todo::

        WRITEME
    """
    pool_len = 0

    # count number of pixels after compression
    for r in rings:
        if (img_h%r) != 0 or (img_w%r) != 0:
            raise ValueError('Image width (%i) or height (%i) is not a multiple of ring size %i' %
                             (img_h, img_w, r))
        pool_len +=  2*img_h/r +  2*(img_w - 2*r)/r
        img_h -= 2*r
        img_w -= 2*r

    return pool_len + img_h * img_w


def encode(topo_X, rings):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    topo_X : WRITEME
        dataset matrix in topological format (batch, rows, cols, chans)
    rings : WRITEME
        list of ring_sizes which were used to generate dense_input
    """
    (batch_size, img_h, img_w, chans) = topo_X.shape

    # determine output shape
    out_size = get_encoded_size(img_h, img_w, rings)
    output = numpy.zeros((batch_size, out_size * chans))

    start_idx = 0
    # perform retina encoding on each channel separately
    for chan_i in xrange(chans):
        channel = topo_X[..., chan_i]
        start_idx = foveate_channel(channel, rings, output, start_idx)

    return output

def decode(dense_X, img_shp, rings):
    """
    .. todo::

        WRITEME

	Parameters
    ----------
    dense_X : WRITEME
        matrix in DenseDesignMatrix format (batch, dim)
    img_shp : WRITEME
        tuple of image dimensions (rows, cols, chans)
    rings : WRITEME
        list of ring_sizes which were used to generate dense_input
    """
    out_shp = [len(dense_X)] + list(img_shp)
    output = numpy.zeros(out_shp)

    start_idx = 0
    # perform retina encoding on each channel separately
    for chan_i in xrange(out_shp[-1]):
        channel = output[..., chan_i]
        start_idx = defoveate_channel(channel, rings, dense_X, start_idx)

    return output


class RetinaEncodingBlock(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    rings : WRITEME
    """

    def __init__(self, rings):
        self.rings = rings

    def perform(self, V):
        assert V.ndim == 4
        return encode(V, self.rings)

    def apply(self, dataset, can_fit=False):
        topo_X = dataset.get_topological_view()
        fov_X = encode(topo_X, self.rings)
        dataset.set_design_matrix(fov_X)


class RetinaDecodingBlock(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    img_shp : WRITEME
    rings : WRITEME
    """
    def __init__(self, img_shp, rings):
        """
        .. todo::

            WRITEME
        """
        self.img_shp = img_shp
        self.rings = rings

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        topo_X = self.perform(X)
        dataset.set_topological_view(topo_X)

    def perform(self, X):
        """
        .. todo::

            WRITEME
        """
        return decode(X, self.img_shp, self.rings)


class RetinaCodingViewConverter(DefaultViewConverter):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    shape : iterable
        List or tuple of three ints: rows, cols, channels
    rings : WRITEME
    """

    def __init__(self, shape, rings):
        self.shape = shape
        self.rings = rings

        rows, cols, channels = shape

        self.topo_space = Conv2DSpace(shape=[rows, cols], num_channels=channels)

        self.decoder = RetinaDecodingBlock(shape, rings)
        self.encoder = RetinaEncodingBlock(rings)

    def design_mat_to_topo_view(self, X):
        """
        .. todo::

            WRITEME
        """
        return self.decoder.perform(X)

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        return self.design_mat_to_topo_view(X)

    def topo_view_to_design_mat(self, V):
        """
        .. todo::

            WRITEME
        """
        return self.encoder.perform(V)
