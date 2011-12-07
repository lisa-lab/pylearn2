import numpy as np

cimport cython
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
cdef int DTYPE_NUM = np.NPY_UINT8

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray rgb_to_gray(np.ndarray[DTYPE_t, ndim=4] video):
    cdef np.ndarray[DTYPE_t, ndim=3] target
    target = np.PyArray_ZEROS(3, video.shape, DTYPE_NUM, 0)
    cdef np.npy_intp i, j, k
    for i in xrange(video.shape[0]):
        for j in xrange(video.shape[1]):
            for k in xrange(video.shape[2]):
                # These are the standard values for the Y channel of
                # the YCvCr color map (the 'luma').`
                target[i, j, k] = <DTYPE_t>(
                                    (<double>video[i, j, k, 0] * 0.2125 +
                                     <double>video[i, j, k, 1] * 0.7154 +
                                     <double>video[i, j, k, 2] * 0.0721)
                )
    return target
