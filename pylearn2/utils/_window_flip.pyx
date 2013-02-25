"""
Routines for extracting (reflected) windows of image stacks in
(channels, rows, cols, batch_size) format.
"""
import numpy as np
cimport numpy as np
cimport cython


cdef extern from "stdlib.h":
    int rand_r(unsigned int *seedp) nogil


def _check_args(np.ndarray[np.float32_t, ndim=4] images,
                tuple window_shape,
                np.ndarray[np.float32_t, ndim=4] out=None):
    """
    Do common validation for the two routines in this file.
    """
    cdef np.npy_intp rows = images.shape[1]
    cdef np.npy_intp cols = images.shape[2]
    cdef np.npy_intp window_r = window_shape[0]
    cdef np.npy_intp window_c = window_shape[1]
    if len(window_shape) != 2:
        raise ValueError("window_shape should be length 2")
    elif window_r > rows or window_c > cols:
        raise ValueError("window_shape (%d, %d) greater than image shape "
                         "(%d, %d)" % (window_r, window_c, rows, cols))
    if out is not None:
        if (out.shape[0] != images.shape[0] or out.shape[1] != window_r or
            out.shape[2] != window_c or out.shape[3] != images.shape[3]):
            raise ValueError("out argument had wrong shape")


@cython.boundscheck(False)
@cython.cdivision(True)
def random_window_and_flip_c01b(np.ndarray[np.float32_t, ndim=4] images,
                                tuple window_shape,
                                np.ndarray[np.float32_t, ndim=4] out=None,
                                object rng=(2013, 2, 20)):
    """
    Transform a stack of images by taking random windows on each
    image and randomly flipping on the horizontal axis.

    Parameters
    ----------
    images : ndarray, 4-dimensional, dtype=float32
        An array of images, shape (channels, rows, cols, batch_size)

    window_shape : tuple
        A length-2 tuple of (window_rows, window_cols) such that
        window_rows <= rows and window_cols <= cols.

    out : ndarray, 4-dimensional, dtype=float32, optional
        An array to use instead of allocating an output buffer,
        of  shape `(channels, window_rows, window_cols, batch_size)`

    rng : `numpy.random.RandomState` or seed, optional
        A random number generator, or a seed used to create one.
        A default seed is used to ensure deterministic behaviour.

    Returns
    -------
    out : ndarray, 4-dimensional, dtype=float32
        An array containing the randomly windowed and flipped images.
        If `out` was provided as an argument, the same array is
        returned.
    """
    cdef np.npy_intp channels = images.shape[0]
    cdef np.npy_intp rows = images.shape[1]
    cdef np.npy_intp cols = images.shape[2]
    cdef np.npy_intp batch = images.shape[3]
    cdef np.npy_intp window_r = window_shape[0]
    cdef np.npy_intp window_c = window_shape[1]
    cdef np.npy_intp offset_r, offset_c, o_j, o_k, example, i, j, k, flip
    cdef unsigned int seed
    cdef np.npy_intp row_offset_max = rows - window_r
    cdef np.npy_intp col_offset_max = cols - window_c
    _check_args(images, window_shape, out)
    if not hasattr(rng, 'random_integers'):
        rng = np.random.RandomState(rng)
    if out is None:
        out = np.empty((channels, window_r, window_c, batch),
                       dtype='float32')
    seed = rng.random_integers(4294967295)
    for example in range(batch):
        offset_r = rand_r(&seed) % (row_offset_max + 1)
        offset_c = rand_r(&seed) % (col_offset_max + 1)
        flip = rand_r(&seed) % 2
        for i in range(channels):
            for j in range(offset_r, offset_r + window_r):
                for k in range(offset_c, offset_c + window_c):
                    o_j = j - offset_r
                    if flip:
                        o_k = window_c - (k - offset_c) - 1
                    else:
                        o_k = k - offset_c
                    out[i, o_j, o_k, example] = images[i, j, k, example]
    return out


@cython.boundscheck(False)
@cython.cdivision(True)
def random_window_and_flip_b01c(np.ndarray[np.float32_t, ndim=4] images,
                                tuple window_shape,
                                np.ndarray[np.float32_t, ndim=4] out=None,
                                object rng=(2013, 2, 20)):
    """
    Transform a stack of images by taking random windows on each
    image and randomly flipping on the horizontal axis.

    Parameters
    ----------
    images : ndarray, 4-dimensional, dtype=float32
        An array of images, shape (batch_size, rows, cols, channels)

    window_shape : tuple
        A length-2 tuple of (window_rows, window_cols) such that
        window_rows <= rows and window_cols <= cols.

    out : ndarray, 4-dimensional, dtype=float32, optional
        An array to use instead of allocating an output buffer,
        of  shape `(batch_size, window_rows, window_cols, channels)`

    rng : `numpy.random.RandomState` or seed, optional
        A random number generator, or a seed used to create one.
        A default seed is used to ensure deterministic behaviour.

    Returns
    -------
    out : ndarray, 4-dimensional, dtype=float32
        An array containing the randomly windowed and flipped images.
        If `out` was provided as an argument, the same array is
        returned.
    """
    cdef np.npy_intp batch = images.shape[0]
    cdef np.npy_intp rows = images.shape[1]
    cdef np.npy_intp cols = images.shape[2]
    cdef np.npy_intp channels = images.shape[3]
    cdef np.npy_intp window_r = window_shape[0]
    cdef np.npy_intp window_c = window_shape[1]
    cdef np.npy_intp offset_r, offset_c, o_j, o_k, example, i, j, k, flip
    cdef unsigned int seed
    cdef np.npy_intp row_offset_max = rows - window_r
    cdef np.npy_intp col_offset_max = cols - window_c
    _check_args(images, window_shape, out)
    if not hasattr(rng, 'random_integers'):
        rng = np.random.RandomState(rng)
    if out is None:
        out = np.empty((batch, window_r, window_c, channels),
                       dtype='float32')
    seed = rng.random_integers(4294967295)
    for example in range(batch):
        offset_r = rand_r(&seed) % (row_offset_max + 1)
        offset_c = rand_r(&seed) % (col_offset_max + 1)
        flip = rand_r(&seed) % 2
        for j in range(offset_r, offset_r + window_r):
            for k in range(offset_c, offset_c + window_c):
                for i in range(channels):
                    o_j = j - offset_r
                    if flip:
                        o_k = window_c - (k - offset_c) - 1
                    else:
                        o_k = k - offset_c
                    out[example, o_j, o_k, i] = images[example, j, k, i]
    return out
