"""
.. todo::

    WRITEME
"""
from pylearn2.datasets import utlc
import numpy as N


class Avicenna(object):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
    standardize : WRITEME
    """

    def __init__(self, which_set, standardize):
        train, valid, test = utlc.load_ndarray_dataset('avicenna')

        if which_set == 'train':
            self.X = train
        elif which_set == 'valid':
            self.X = valid
        elif which_set == 'test':
            self.X = test
        else:
            assert False

        if standardize:
            union = N.concatenate([train, valid, test], axis=0)
            # perform mean and std in float64 to avoid losing
            # too much numerical precision
            self.X -= union.mean(axis=0, dtype='float64')
            std = union.std(axis=0, dtype='float64')
            std[std < 1e-3] = 1e-3
            self.X /= std

    def get_design_matrix(self):
        """
        .. todo::

            WRITEME
        """
        return self.X
