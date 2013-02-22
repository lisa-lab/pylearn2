"""
Parallelized k-means module.
"""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from numpy.random import normal
cdef extern from "numpy/npy_math.h":
    double NPY_INFINITY

# TODO: Allow this to accept float32 or float32 using
# Cython's fused types, coming in 0.16.

ctypedef np.float32_t DTYPE_t

# Determine the right dtype for arrays of indices at compile time.
IF UNAME_MACHINE[-2:] == '64':
    ctypedef np.int64_t INTP_t
    intp = np.int64
ELSE:
    ctypedef np.int32_t INTP_t
    intp = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _compute_means(np.ndarray[DTYPE_t, ndim=2] data,
                                np.ndarray[INTP_t, ndim=1] assign,
                                np.ndarray[DTYPE_t, ndim=2] means,
                                np.ndarray[INTP_t, ndim=1] counts):
    """
    Compute the new centroids given the assignments in `assign`,
    leaving the results in `means`.

    Parameters
    ----------
    data : ndarray, 2-dimensional, float32
        Matrix of features with training examples indexed along
        the first dimension and features indexed along the second.
    assign : ndarray, 1-dimensional, int32/int64 (platform dependent)
        A vector of length `data.shape[0]` containing an index into
        `means`, indicating to which centroid a training example is
        assigned.
    means : ndarray, 2-dimensional, float32
        A matrix of shape `(k, data.shape[0])`, with each row
        representing a centroid vector. This array be overwritten
        by this function.
    counts : ndarray, 2-dimensional, int32/int64 (platform dependent)
        A vector of length `k` indicating the number of training
        examples assigned to each centroid. This array wil be
        overwritten by this function.

    Notes
    -----
    The data in `counts` argument at call time is never actually
    used, it is simply made an argument to this function to avoid
    reallocating a new buffer on every mean computation (which can
    be a slight performance hit if the number of centroids is
    substantial).

    This parallelizes over features (columns of `data` and
    `means`) using OpenMP via Cython's `cython.parallel.prange`.
    Parallelizing over examples would also be possible but would
    result in slightly different results compared with a non-parallel
    version due to the non-associativity of floating point addition.

    Cython currently gives the warning "buffer unpacking not
    optimized away" due to this being an inline function. This should
    be fixed in the next release, at which point we will reap the
    full benefits of inlining this.
    """
    # Convenience variables and loop indices.
    cdef np.npy_intp ndata = data.shape[0]
    cdef np.npy_intp nfeat = data.shape[1]
    cdef np.npy_intp k = means.shape[0]
    cdef np.npy_intp example, feature, centroid
    # Zero the counts vector before repopulating it.
    for centroid in range(k):
        counts[centroid] = 0
    # Count the number of times each centroid occurs in the assignments.
    for example in range(ndata):
        counts[assign[example]] += 1
    # Main worker loop: for each feature, start by zeroing its value
    # for every centroid, then compute the sum of all examples assigned
    # to it, and finally normalize.
    for feature in prange(nfeat, nogil=True):
        for centroid in range(k):
            # If a centroid has no points assigned to it, leave it alone.
            if counts[centroid] > 0:
                means[centroid, feature] = 0.
        for example in range(ndata):
            means[assign[example], feature] += data[example, feature]
        for centroid in range(k):
            # Only normalize if counts[centroid] is non-zero to avoid NaN.
            if counts[centroid] > 0:
                means[centroid, feature] /= counts[centroid]
            else:
                means[centroid, feature] = 0.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef tuple kmeans(np.ndarray[DTYPE_t, ndim=2] data, np.npy_intp k,
                   np.npy_intp max_iter=1000, np.ndarray init=None,
                   rng=(2013, 02, 22)):
    """
    Run k-means on a dense matrix of features, parallelizing
    computations with OpenMP and BLAS where possible.

    Parameters
    ----------
    data : ndarray, 2-dimensional, float32
        Matrix of features with training examples indexed along
        the first dimension and features indexed along the second.
    k : int
        Number of centroids ("means") to use.
    max_iter : int, optional
        Maximum number of iterations of the algorithm to run
        (default is 1000).
    init : ndarray, 2-dimensional, float32, optional
        An initial set of centroids to use instead of the default
        initialization. This array **must** be of shape
        `(k, data.shape[1])` if it is provided, and **will be
        overwritten**.
    rng : RandomState object or seed, optional
        A random number generator instance to use for initialization
        in the absence of `init`, or a seed with which to create one.
        See the docstring for `numpy.random.RandomState` for
        details on the accepted seed formats. Default is a
        deterministic seed to enusre reproducibility.

    Returns
    -------
    means : ndarray, 2-dimensional, float32
        A matrix of shape `(k, data.shape[0])`, with each row
        representing a centroid vector. If `init` was provided, this
        will be the exact same array, but with the contents replaced
        with the values of the centroids after k-means has terminated.
    assign : ndarray, 1-dimensional, int32/int64 (platform dependent)
        A vector with one entry per training example, indicating
        the index of the closest centroid at termination.
    iteration : int
        The number of iterations of k-means actually performed. This
        will be less than or equal to `max_iter` specified in the
        input arguments.
    converged : boolean
        A boolean flag indicating whether or not the algorithm
        converged (i.e. False if the assignments changed in the
        last iteration). This disambiguates the rare but feasible
        case where convergence took place just as `max_iter` was
        reached.

    Notes
    -----
    The main bottleneck of k-means is the distance matrix computation.
    This implementation uses `numpy.dot` for this, so you should
    ensure that your installation of NumPy is linked against a good
    multithreaded BLAS implementation for optimal performance. If
    NumPy is linked against the Intel Math Kernel Library (as it will
    be if you are using the full version of the Enthought Python
    Distribution), make sure the environment variable `MKL_NUM_THREADS`
    is set to the number of cores you wish it to use.

    Significant gains can be made by parallelizing the quantization
    and centroid computation as well. This implementation uses OpenMP
    to parallelize mean computation over *features* (columns of
    the data matrix) and quantization over *training examples*. Make
    sure `OMP_NUM_THREADS` is set to the desired number of worker
    threads/CPU cores.
    """
    cdef np.npy_intp ndata = data.shape[0]
    cdef np.npy_intp nfeat = data.shape[1]
    dtype = data.dtype
    cdef np.ndarray[DTYPE_t, ndim=1] mindist = np.empty(ndata, dtype=dtype)
    cdef np.ndarray[INTP_t, ndim=1] counts = np.empty(k, dtype=intp)
    # Allocate space for the assignment indices, distance matrix, the means.
    cdef np.ndarray[DTYPE_t, ndim=2] dists = np.empty((ndata, k), dtype=dtype)
    cdef np.ndarray[DTYPE_t, ndim=1] m_sqnorm = np.empty(k, dtype=dtype)
    cdef np.ndarray[DTYPE_t, ndim=2] means
    # Declare variables for the current assignments and current argmin.
    # Storing the current argmin separately lets us easily check for
    # convergence at a memory cost of (pointer width * ndata).
    cdef np.ndarray[INTP_t, ndim=1] assign
    cdef np.ndarray[INTP_t, ndim=1] argmin = np.empty(ndata, dtype=intp)
    cdef np.npy_intp example, centroid, feature
    if init is not None:
        if rng is not None:
            raise ValueError('rng argument unused if init is provided')
        if init.shape[0] != k or init.shape[1] != nfeat:
            raise ValueError('init if provided must have shape (k, '
                             'data.shape[1])')
        means = init
        assign = np.empty(ndata, dtype=intp)
    else:
        means = np.empty((k, nfeat), dtype=dtype)
        # Randomly initialize assignments to uniformly drawn training points.
        if not hasattr(rng, 'random_integers'):
            rng = np.random.RandomState(rng)
        assign = rng.random_integers(0, k - 1, size=ndata).astype(intp)
        # Compute the means from the random initial assignments.
        _compute_means(data, assign, means, counts)

    for iteration in range(max_iter):
        # Quantization step: compute squared distance between every point
        # and every mean.
        # The distance between each of the data points and each of the means
        # can be computed by a matrix product (times -2) plus squared norms.
        np.dot(data, means.T, out=dists)

        # Compute the squared norm of each of the centroids (necessary
        # for determining relative distances below).
        for centroid in prange(k, nogil=True):
            m_sqnorm[centroid] = 0.
            for feature in range(nfeat):
                m_sqnorm[centroid] += (means[centroid, feature] *
                                       means[centroid, feature])

        # Determine the minimum distance cluster to each example. Note that
        # we are actually determining max(m'x - 0.5 * m'm) which is equivalent
        # to min(x'x - 2 * m'x + m'm), since the first term never changes.
        for example in prange(ndata, nogil=True):
            # Initialize the min and argmin to invalid values.
            argmin[example] = -1
            mindist[example] = -NPY_INFINITY
            for centroid in range(k):
                dists[example, centroid] -= 0.5 * m_sqnorm[centroid]
                if dists[example, centroid] > mindist[example]:
                    mindist[example] = dists[example, centroid]
                    argmin[example] = centroid

        # Check previous assignment against current assignment to determine if
        # we've converged.  NOTE: Don't do this with prange as sometimes the
        # variable will not get set correctly.
        # TODO: Do this check in the parallel loop above using a with
        # parallel() block.
        converged = True
        for example in range(ndata):
            if argmin[example] != assign[example]:
                converged = False
            assign[example] = argmin[example]
        # If the assignment has changed, recompute means and continue the loop.
        if not converged:
            _compute_means(data, assign, means, counts)
        else:
            break
    return assign, means, iteration + 1, converged
