"""
The four regions task. A synthetic dataset from the late 1980s,
a 4-class classification problem.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"


import numpy as np
from theano import config
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils.rng import make_np_rng


def _four_regions_labels(points):
    """
    Returns labels for points in [-1, 1]^2 from the
    "four regions" benchmark task first described by
    Singhal and Wu.

    Parameters
    ----------
    points : array_like, 2-dimensional
        An (N, 2) list of 2-dimensional points.

    Returns
    -------
    labels : ndarray, 1-dimensional
        An N-length array of labels in [1, 2, 3, 4].

    References
    ----------
    .. [1] S. Singhal and L. Wu, "Training multilayer perceptrons
      with the extended Kalman algorithm". Advances in Neural
      Information Processing Systems, 1, (1988) pp 133-140.
      http://books.nips.cc/papers/files/nips01/0133.pdf
    """
    points = np.asarray(points)
    region = np.zeros(points.shape[0], dtype='uint8')
    tophalf = points[:, 1] > 0
    righthalf = points[:, 0] > 0
    dists = np.sqrt(np.sum(points ** 2, axis=1))

    # The easy ones -- the outer shelf.
    region[dists > np.sqrt(2)] = 255
    outer = dists > 5. / 6.
    region[np.logical_and(tophalf, outer)] = 3
    region[np.logical_and(np.logical_not(tophalf), outer)] = 0

    firstring = np.logical_and(dists > 1. / 6., dists <= 1. / 2.)
    secondring = np.logical_and(dists > 1. / 2., dists <= 5. / 6.)

    # Region 2 -- right inner and left outer, excluding center nut
    region[np.logical_and(firstring, righthalf)] = 2
    region[np.logical_and(secondring, np.logical_not(righthalf))] = 2

    # Region 1 -- left inner and right outer, including center nut
    region[np.logical_and(secondring, righthalf)] = 1
    region[np.logical_and(np.logical_not(righthalf), dists < 1. / 2.)] = 1
    region[np.logical_and(righthalf, dists < 1. / 6.)] = 1
    assert np.all(region >= 0) and np.all(region <= 3)
    return region


class FourRegions(DenseDesignMatrix):

    """
    Constructs a dataset based on the four regions
    benchmark by sampling random uniform points in [-1, 1]^2
    and constructing the label.

    Parameters
    ----------
    num_examples : int
        The number of examples to generate.

    rng : RandomState or seed
        A random number generator or a seed used to construct it.

    References
    ----------
    .. [1] S. Singhal and L. Wu, "Training multilayer perceptrons
      with the extended Kalman algorithm". Advances in Neural
      Information Processing Systems, 1, (1988) pp 133-140.
      http://books.nips.cc/papers/files/nips01/0133.pdf
    """
    _default_seed = (2013, 05, 17)

    def __init__(self, num_examples, one_hot=False, rng=(2013, 05, 17)):
        """
        .. todo::

            WRITEME
        """
        rng = make_np_rng(rng, self._default_seed, which_method='uniform')
        X = rng.uniform(-1, 1, size=(num_examples, 2))
        if not one_hot:
            y = _four_regions_labels(X)
        else:
            y = np.zeros((num_examples, 4), dtype=config.floatX)
            labels = _four_regions_labels(X)
            y.flat[np.arange(0, 4 * num_examples, 4) + labels] = 1.
        super(FourRegions, self).__init__(X=X, y=y)
