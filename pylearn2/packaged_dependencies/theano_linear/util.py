
from imaging import tile_slices_to_image

_ndarray_status_fmt='%(msg)s shape=%(shape)s min=%(min)f max=%(max)f'


def ndarray_status(x, fmt=_ndarray_status_fmt, msg="", **kwargs):
    kwargs.update(dict(
            msg=msg,
            min=x.min(),
            max=x.max(),
            mean=x.mean(),
            var = x.var(),
            shape=x.shape))
    return fmt%kwargs

