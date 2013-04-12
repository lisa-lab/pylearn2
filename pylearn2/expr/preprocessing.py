"""
Low-level utilities for preprocessing. Should be functions that apply
to NumPy arrays, not preprocessor classes (though preprocessor classes
should reuse these).
"""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"

import numpy

def global_contrast_normalize(X, scale=1., subtract_mean=True, std_norm=False,
                              add_const=0., clip_below=1e-8):
    """
    Global contrast normalizes by subtracting the mean across
    features and then projects onto the unit sphere, optionally
    scaling by a const.

    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    std_norm : float, optional
        Normalize by the (adjusted, see `eps` below) per-example standard
        deviation across features instead of vector norm. Corresponds to
        a sum in place of a mean.

    add_const : float, optional
        Fudge factor added inside the square root. Defaults to 0.

    clip_below : float, optional
        Clip the divisors to this minimum value.
    """
    scale = float(scale)
    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()
    if std_norm:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        normalizers = numpy.sqrt(add_const + X.var(axis=1, ddof=1)) / scale
    else:
        normalizers = numpy.sqrt(add_const + (X ** 2).sum(axis=1)) / scale
    # Don't normalize by anything too small.
    normalizers[normalizers < add_const] = add_const
    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X
