"""Utilities for manipulating binary strings/masks."""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"

import numpy as np


def all_bit_strings(bits, dtype='uint8'):
    """
    Create a matrix of all binary strings of a given width as the rows.

    Parameters
    ----------
    bits : int
        The number of bits to count through.

    dtype : str or dtype object
        The dtype of the returned array.

    Returns
    -------
    bit_strings : ndarray, shape (2 ** bits, bits)
        The numbers from 0 to 2 ** bits - 1 as binary numbers, most
        significant bit first.

    Notes
    -----
    Obviously the memory requirements of this are exponential in the first
    argument, so use with caution.
    """
    return np.array([map(int, np.binary_repr(i, width=bits))
                     for i in xrange(0, 2 ** bits)], dtype=dtype)
