import theano.tensor as T

def identity(x):
    """
    Importable identity function. Created for the purposes of pickling.
    """
    return x

def relu(x):
    return T.max(0, x)

def plushmax(x, eps=0.0, min_val=0.0):
    """
    A softer softmax.

    Instead of computing exp(x_i) / sum_j(exp(x_j)), this computes
    (exp(x_i) + eps) / sum_j(exp(x_j) + eps)

    Additionally, all values in the return vector will be at least min_val.
    eps may be increased to satisfy this constraint.
    """
    assert eps >= 0.0

    MIN_SAFE = max(0.00001, min_val)

    s = T.sum(T.exp(x), axis=1, keepdims=True)
    safe_eps = (MIN_SAFE * s) / (1.0 - x.shape[1] * MIN_SAFE)
    safe_eps = T.cast(safe_eps, theano.config.floatX)

    eps = T.maximum(eps, safe_eps)

    y = x + T.log(1.0 + eps * T.exp(-x))
    return T.nnet.softmax(y)
