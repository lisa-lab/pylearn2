"""
.. todo::

    WRITEME
"""
import theano
T = theano.tensor


def identity(x):
    """
    .. todo::

        WRITEME properly

    Importable identity function. Created for the purposes of pickling.
    """
    return x


def relu(x):
    """
    .. todo::

        WRITEME properly

    Rectified linear activation
    """
    return T.max(0, x)


def _rescale_softmax(sm, min_val):
    """
    .. todo::

        WRITEME
    """
    n_classes = sm.shape[-1]
    # Avoid upcast to float64 when floatX==float32 and n_classes is int64
    n_classes = n_classes.astype(theano.config.floatX)
    return sm * (1 - n_classes * min_val) + min_val


def rescaled_softmax(x, min_val=1e-5):
    """
    .. todo::

        WRITEME
    """
    return _rescale_softmax(T.nnet.softmax(x), min_val=min_val)
