"""
.. todo::

    WRITEME
"""
import numpy
import theano
T = theano.tensor


def log_mean_exp(a):
    """
    .. todo::

        WRITEME
    """
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))

def make_lpdf(mu, sigma):
    """
    Makes a Theano function that allows the evalution of a Parzen windows
    estimator (aka kernel density estimator) where the Kernel is a normal
    distribution with stddev sigma and with points at mu.

    Parameters
    -----------
    mu : numpy matrix
        Contains the data points over which this distribution is based.
    sigma : scalar
        The standard deviation of the normal distribution around each data \
        point.

    Returns
    -------
    lpdf : callable
        Estimator of the log of the probability density under a point.
    """
    x = T.matrix()
    mu = theano.shared(mu)

    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma

    E = log_mean_exp(-0.5*(a**2).sum(2))

    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


class ParzenWindows(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    samples : numpy matrix
        See description for make_lpdf
    sigma : scalar
        See description for make_lpdf
    """
    def __init__(self, samples, sigma):
        # just keeping these for debugging/examination, not needed
        self._samples = samples
        self._sigma = sigma

        self.lpdf = make_lpdf(samples, sigma)

    def get_ll(self, x, batch_size=10):
        """
        Evaluates the log likelihood of a set of datapoints with respect to the
        probability distribution.

        Parameters
        ----------
        x : numpy matrix
            The set of points for which you want to evaluate the log \
            likelihood.
        """
        inds = range(x.shape[0])
        n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

        lls = []
        for i in range(n_batches):
            lls.extend(self.lpdf(x[inds[i::n_batches]]))

        return numpy.array(lls).mean()
