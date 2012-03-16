from theano.tensor.nnet.conv import conv2d, ConvOp

from .imaging import tile_slices_to_image, most_square_shape
from .linear import LinearTransform
import numpy
from theano import tensor

def tile_conv_weights(w, flip=False, scale_each=False):
    """
    Return something that can be rendered as an image to visualize the filters.
    """
    if w.shape[1] != 3:
        raise NotImplementedError('not rgb', w.shape)
    if w.shape[2] != w.shape[3]:
        raise NotImplementedError('not square', w.shape)
    wmin, wmax = w.min(), w.max()
    if not scale_each:
        w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    trows, tcols= most_square_shape(w.shape[0])
    outrows = trows * w.shape[2] + trows-1
    outcols = tcols * w.shape[3] + tcols-1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')

    tr_stride= 1+w.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            # this is supposed to flip the filters back into the image
            # coordinates as well as put the channels in the right place, but I
            # don't know if it really does that
            tmp = w[tr*tcols+tc].transpose(1,2,0)[
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            if scale_each:
                tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    return out


class Conv2d(LinearTransform):
    """
    XXX
    """

    def __init__(self, filters, img_shape, subsample=(1,1), border_mode='valid',
            filters_shape=None, message=""):
        super(Conv2d, self).__init__([filters])
        self._filters = filters
        if filters_shape is None:
            self._filters_shape = tuple(filters.get_value().shape)
        else:
            self._filters_shape = tuple(filters_shape)
        self._img_shape = tuple(img_shape)
        self._subsample = tuple(subsample)
        self._border_mode = border_mode
        if message:
            self._message = message
        else:
            self._message = filters.name
        if not len(self._img_shape)==4:
            raise TypeError('need 4-tuple shape', self._img_shape)
        if not len(self._filters_shape)==4:
            raise TypeError('need 4-tuple shape', self._filters_shape)

    def lmul(self, x):
        # dot(x, A)
        return conv2d(
                x, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )

    def lmul_T(self, x):
        # dot(x, A.T)
        dummy_v = tensor.tensor4()
        z_hs = conv2d(dummy_v, self._filters,
                image_shape=self._img_shape,
                filter_shape=self._filters_shape,
                subsample=self._subsample,
                border_mode=self._border_mode,
                )
        xfilters, xdummy = z_hs.owner.op.grad((dummy_v, self._filters), (x,))
        return xfilters

    def row_shape(self):
        return self._img_shape[1:]

    def col_shape(self):
        rows_cols = ConvOp.getOutputShape(
                self._img_shape[2:],
                self._filters_shape[2:],
                self._subsample,
                self._border_mode)
        rval = (self._filters_shape[0],)+tuple(rows_cols)
        return rval

    def tile_columns(self, scale_each=True, **kwargs):
        return tile_slices_to_image(
                self._filters.get_value()[:,:,::-1,::-1].transpose(0,2,3,1),
                scale_each=scale_each,
                **kwargs)

    def print_status(self):
        raise NotImplementedError('TODO fix broken method')
        #print ndarray_status(
        #        self._filters.get_value(borrow=True),
        #        msg='Conv2d{%s}'%self._message)

