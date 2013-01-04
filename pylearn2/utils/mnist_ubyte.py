"""Low-level utilities for reading in raw MNIST files."""

__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import struct
import numpy

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


class open_if_filename(object):
    def __init__(self, f, mode='r', buffering=-1):
        self._f = f
        self._mode = mode
        self._buffering = buffering
        self._handle = None

    def __enter__(self):
        if isinstance(self._f, basestring):
            self._handle = open(self._f, self._mode, self._buffering)
        else:
            self._handle = self._f
        return self._handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not self._f:
            self._handle.close()


def read_mnist_images(fn, dtype=None):
    """
    Read MNIST images from the original ubyte file format.

    Parameters
    ----------
    fn : str or object
        Filename/path from which to read labels, or an open file
        object for the same (will not be closed for you).

    dtype : str or object, optional
        A NumPy dtype or string that can be converted to one.
        If unspecified, images will be returned in their original
        unsigned byte format.

    Returns
    -------
    images : ndarray, shape (n_images, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----
    If the dtype provided was boolean, the resulting array will
    be boolean with `True` if the corresponding pixel had a value
    greater than or equal to 128, `False otherwise.

    If the dtype provided was a float or complex dtype, the values
    will be mapped to the unit interval [0, 1], with pixel values
    that were 255 in the original unsigned byte representation
    equal to 1.0.
    """
    with open_if_filename(fn, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError('wrong magic number reading MNIST image file: ' +
                             fn)
        array = numpy.fromfile(f, dtype='uint8').reshape((number, rows, cols))
    if dtype:
        dtype = numpy.dtype(dtype)
        # If the user wants booleans, threshold at half the range.
        if dtype.kind is 'b':
            array = array >= 128
        else:
            # Otherwise, just convert.
            array = array.astype(dtype)
        # I don't know why you'd ever turn MNIST into complex,
        # but just in case, check for float *or* complex dtypes.
        # Either way, map to the unit interval.
        if dtype.kind in ('f', 'c'):
            array /= 255.
    return array


def read_mnist_labels(fn):
    """
    Read MNIST labels from the original ubyte file format.

    Parameters
    ----------
    fn : str or object
        Filename/path from which to read labels, or an open file
        object for the same (will not be closed for you).

    Returns
    -------
    labels : ndarray, shape (nlabels,)
        A one-dimensional unsigned byte array containing the
        labels as integers.
    """
    with open_if_filename(fn, 'rb') as f:
        magic, number = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError('wrong magic number reading MNIST label file: ' +
                             fn)
        array = numpy.fromfile(f, dtype='uint8')
    return array
