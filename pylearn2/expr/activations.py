import theano
T = theano.tensor

def identity(x):
    """
    Importable identity function. Created for the purposes of pickling.
    """
    return x

def relu(x):
    """
    Rectified linear activation
    """
    return T.max(0, x)

def _rescale_softmax(sm, min_val):
    n_classes = sm.shape[-1]
    # Avoid upcast to float64 when floatX==float32 and n_classes is int64
    n_classes = n_classes.astype(theano.config.floatX)
    return sm * (1 - n_classes * min_val) + min_val

def rescaled_softmax(x, min_val=1e-5):
    return _rescale_softmax(T.nnet.softmax(x), min_val=min_val)

