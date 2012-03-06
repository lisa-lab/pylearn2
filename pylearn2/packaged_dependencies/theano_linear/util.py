
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


# XXX: copy-paste out of pylearn
try:
    from pylearn.io.image_tiling import tile_slices_to_image
except ImportError:
    def tile_slices_to_image(*args, **kwargs):
        raise NotImplementedError()


