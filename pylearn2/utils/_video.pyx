import numpy as np

cimport cython
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
cdef int DTYPE_NUM = np.NPY_UINT8

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray rgb_to_gray(np.ndarray[DTYPE_t, ndim=4] video):
    """
    rgb_to_gray(video)

    Converts a color video stored as a 4-dimensional uint8 array to
    grayscale using standard YCvCr weighting.

    Parameters
    ----------
    video : ndarray, 4-dimensional, uint8
        Dimensions are (time, row, column, channel). Along the last axis,
        channels are assumed to be ordered (R, G, B).

    Returns
    -------
    out : ndarray, 3-dimensional, uint8
        Same dimensions as video with the RGB channel dimension removed.
    """
    cdef np.ndarray[DTYPE_t, ndim=3] out
    out = np.PyArray_ZEROS(3, video.shape, DTYPE_NUM, 0)
    cdef np.npy_intp i, j, k
    for i in xrange(video.shape[0]):
        for j in xrange(video.shape[1]):
            for k in xrange(video.shape[2]):
                # These are the standard values for the Y channel of
                # the YCvCr color map (the 'luma').`
                out[i, j, k] = <DTYPE_t>(
                                    (<double>video[i, j, k, 0] * 0.2125 +
                                     <double>video[i, j, k, 1] * 0.7154 +
                                     <double>video[i, j, k, 2] * 0.0721)
                )
    return out
