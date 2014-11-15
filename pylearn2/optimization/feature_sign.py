"""
L1-penalized minimization using the feature sign search algorithm.
"""

__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2011, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

__all__ = ["feature_sign_search"]

from itertools import count

import logging
import numpy as np
from theano.compat import six
from theano.compat.six.moves import zip as izip

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _feature_sign_checkargs(dictionary, signals, sparsity, max_iter,
                            solution):
    """
    .. todo::

        WRITEME
    """
    if solution is not None:
        assert solution.ndim == signals.ndim, (
            "if provided, solutions must be same number of dimensions as "
            "signals"
        )
    if signals.ndim == 1:
        assert signals.shape[0] == dictionary.shape[0], (
            "signals.ndim == 1, but signals.shape[0] !=  dictionary.shape[0]"
        )
        if solution is not None:
            assert solution.shape[0] == dictionary.shape[1], (
                "solutions array is wrong shape (ndim=1, should have first "
                "dimension %d given dictionary)" % dictionary.shape[1]
            )
    elif signals.ndim == 2:
        assert signals.shape[1] == dictionary.shape[0], (
            "signals.ndim == 2, but signals.shape[1] !=  dictionary.shape[0]"
        )
        if solution is not None:
            assert solution.shape[0] == signals.shape[0], (
                "solutions array is wrong shape (ndim=2, should have first "
                "dimension %d given signals)" % signals.shape[0]
            )
            assert solution.shape[1] == dictionary.shape[1], (
                "solutions array is wrong shape (ndim=1, should have second "
                "dimension %d given dictionary)" % dictionary.shape[1]
            )


def _feature_sign_search_single(dictionary, signal, sparsity, max_iter,
                                solution=None):
    """
    Solve a single L1-penalized minimization problem with
    feature-sign search.

    Parameters
    ----------
    dictionary : array_like, 2-dimensional
        The dictionary of basis functions from which to form the
        sparse linear combination.
    signal : array_like, 1-dimensional
        The signal being decomposed as a sparse linear combination
        of the columns of the dictionary.
    sparsity : float
        The coefficient on the L1 penalty term of the cost function.
    max_iter : int
        The maximum number of iterations to run.
    solution : ndarray, 1-dimensional, optional
        Pre-allocated vector to use to store the solution.

    Returns
    -------
    solution : ndarray, 1-dimensional
        Vector containing the solution. If an array was passed in
        as the argument `solution`, it will be updated in place
        and the same object will be returned.
    count : int
        The number of iterations of the algorithm that were run.

    Notes
    -----
    See the docstring of `feature_sign_search` for details on the
    algorithm.
    """
    # This prevents the sparsity penalty scalar from upcasting the entire
    # rhs vector sent to linalg.solve().
    sparsity = np.array(sparsity).astype(dictionary.dtype)
    effective_zero = 1e-18
    # precompute matrices for speed.
    gram_matrix = np.dot(dictionary.T, dictionary)
    target_correlation = np.dot(dictionary.T, signal)
    # initialization goes here.
    if solution is None:
        solution = np.zeros(gram_matrix.shape[0], dtype=dictionary.dtype)
    else:
        assert solution.ndim == 1, "solution must be 1-dimensional"
        assert solution.shape[0] == dictionary.shape[1], (
            "solution.shape[0] does not match dictionary.shape[1]"
        )
        # Initialize all elements to be zero.
        solution[...] = 0.
    signs = np.zeros(gram_matrix.shape[0], dtype=np.int8)
    active_set = set()
    z_opt = np.inf
    # Used to store whether max(abs(grad[nzidx] + sparsity * signs[nzidx]))
    # is approximately 0.
    # Set to True here to trigger a new feature activation on first iteration.
    nz_optimal = True
    # second term is zero on initialization.
    grad = - 2 * target_correlation  # + 2 * np.dot(gram_matrix, solution)
    max_grad_zero = np.argmax(np.abs(grad))
    # Just used to compute exact cost function.
    sds = np.dot(signal.T, signal)
    counter = count(0)
    while z_opt > sparsity or not nz_optimal:
        if six.next(counter) == max_iter:
            break
        if nz_optimal:
            candidate = np.argmax(np.abs(grad) * (signs == 0))
            log.debug("candidate feature: %d" % candidate)
            if grad[candidate] > sparsity:
                signs[candidate] = -1.
                solution[candidate] = 0.
                log.debug("added feature %d with negative sign" %
                          candidate)
                active_set.add(candidate)
            elif grad[candidate] < -sparsity:
                signs[candidate] = 1.
                solution[candidate] = 0.
                log.debug("added feature %d with positive sign" %
                          candidate)
                active_set.add(candidate)
            if len(active_set) == 0:
                break
        else:
            log.debug("Non-zero coefficient optimality not satisfied, "
                      "skipping new feature activation")
        indices = np.array(sorted(active_set))
        restr_gram = gram_matrix[np.ix_(indices, indices)]
        restr_corr = target_correlation[indices]
        restr_sign = signs[indices]
        rhs = restr_corr - sparsity * restr_sign / 2
        # TODO: implement footnote 3.
        #
        # If restr_gram becomes singular, check if rhs is in the column
        # space of restr_gram.
        #
        # If so, use the pseudoinverse instead; if not, update to first
        # zero-crossing along any direction in the null space of restr_gram
        # such that it has non-zero dot product with rhs (how to choose
        # this direction?).
        new_solution = np.linalg.solve(np.atleast_2d(restr_gram), rhs)
        new_signs = np.sign(new_solution)
        restr_oldsol = solution[indices]
        sign_flips = np.where(abs(new_signs - restr_sign) > 1)[0]
        if len(sign_flips) > 0:
            best_obj = np.inf
            best_curr = None
            best_curr = new_solution
            best_obj = (sds + (np.dot(new_solution,
                                      np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr))
                        + sparsity * abs(new_solution).sum())
            if log.isEnabledFor(logging.DEBUG):
                # Extra computation only done if the log-level is
                # sufficient.
                ocost = (sds + (np.dot(restr_oldsol,
                                       np.dot(restr_gram, restr_oldsol))
                        - 2 * np.dot(restr_oldsol, restr_corr))
                        + sparsity * abs(restr_oldsol).sum())
                cost = (sds + np.dot(new_solution,
                                     np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr)
                        + sparsity * abs(new_solution).sum())
                log.debug("Cost before linesearch (old)\t: %e" % ocost)
                log.debug("Cost before linesearch (new)\t: %e" % cost)
            else:
                ocost = None
            for idx in sign_flips:
                a = new_solution[idx]
                b = restr_oldsol[idx]
                prop = b / (b - a)
                curr = restr_oldsol - prop * (restr_oldsol - new_solution)
                cost = sds + (np.dot(curr, np.dot(restr_gram, curr))
                              - 2 * np.dot(curr, restr_corr)
                              + sparsity * abs(curr).sum())
                log.debug("Line search coefficient: %.5f cost = %e "
                          "zero-crossing coefficient's value = %e" %
                          (prop, cost, curr[idx]))
                if cost < best_obj:
                    best_obj = cost
                    best_prop = prop
                    best_curr = curr
            log.debug("Lowest cost after linesearch\t: %e" % best_obj)
            if ocost is not None:
                if ocost < best_obj and not np.allclose(ocost, best_obj):
                    log.debug("Warning: objective decreased from %e to %e" %
                              (ocost, best_obj))
        else:
            log.debug("No sign flips, not doing line search")
            best_curr = new_solution
        solution[indices] = best_curr
        zeros = indices[np.abs(solution[indices]) < effective_zero]
        solution[zeros] = 0.
        signs[indices] = np.int8(np.sign(solution[indices]))
        active_set.difference_update(zeros)
        grad = - 2 * target_correlation + 2 * np.dot(gram_matrix, solution)
        z_opt = np.max(abs(grad[signs == 0]))
        nz_opt = np.max(abs(grad[signs != 0] + sparsity * signs[signs != 0]))
        nz_optimal = np.allclose(nz_opt, 0)

    return solution, min(six.next(counter), max_iter)


def feature_sign_search(dictionary, signals, sparsity, max_iter=1000,
                        solution=None):
    """
    Solve L1-penalized quadratic minimization problems with
    feature-sign search.

    Employs the feature sign search algorithm of Lee et al (2006)
    to solve an L1-penalized quadratic optimization problem as a
    sequence of unconstrained quadratic minimization problems over
    subsets of the variables, with candidates for non-zero elements
    chosen by means of a gradient-based criterion.

    Parameters
    ----------
    dictionary : array_like, 2-dimensional
        The dictionary of basis functions from which to form the
        sparse linear combination. Each column constitutes a basis
        vector for the sparse code. There should be as many rows as
        input dimensions in the signal.
    signals : array_like, 1- or 2-dimensional
        The signal(s) to be decomposed as a sparse linear combination
        of the columns of the dictionary. If 2-dimensional, each
        different signal (training case) should be a row of this matrix.
    sparsity : float
        The coefficient on the L1 penalty term of the cost function.
    max_iter : int, optional
        The maximum number of iterations to run, per code vector, if
        the optimization has still not converged. Default is 1000.
    solution : ndarray, 1- or 2-dimensional, optional
        Pre-allocated vector or matrix used to store the solution(s).
        If provided, it should have the same rank as `signals`. If
        2-dimensional, it should have as many rows as `signals`.

    Returns
    -------
    solution : ndarray, 1- or 2-dimensional
        Matrix where each row contains the solution corresponding to a
        row of `signals`. If an array was passed in as the argument
        `solution`, it  will be updated in place and the same object
        will be returned.

    Notes
    -----
    It might seem more natural, from a linear-algebraic point of
    view, to think of both `signals` and `solution` as matrices with
    training examples contained as column vectors; then the overall
    cost function being minimized is

    .. math::
        (Y - AX)^2 + \gamma \sum_{i,j} |X_{ij}|

    with :math:`$A$` representing the dictionary, :math:`Y` being
    `signals.T` and math:`X` being `solutions.T`. However, in order
    to maintain the convention of training examples being indexed
    along the first dimension in the case of 2-dimensional `signals`
    input (as well as provide faster computation via memory locality
    in the case of C-contiguous inputs), this function expects and
    returns input with training examples as rows of a matrix.

    References
    ----------
    .. [1] H. Lee, A. Battle, R. Raina, and A. Y. Ng. "Efficient
       sparse coding algorithms". Advances in Neural Information
       Processing Systems 19, 2007.
    """
    dictionary = np.asarray(dictionary)
    _feature_sign_checkargs(dictionary, signals, sparsity, max_iter, solution)
    # Make things the code a bit simpler by always forcing the
    # 2-dimensional case.
    signals_ndim = signals.ndim
    signals = np.atleast_2d(signals)
    if solution is None:
        solution = np.zeros((signals.shape[0], dictionary.shape[1]),
                            dtype=signals.dtype)
        orig_sol = None
    else:
        orig_sol = solution
        solution = np.atleast_2d(solution)
    # Solve each minimization in sequence.
    for row, (signal, sol) in enumerate(izip(signals, solution)):
        _, iters = _feature_sign_search_single(dictionary, signal, sparsity,
                                               max_iter, sol)
        if iters >= max_iter:
            log.warning("maximum number of iterations reached when "
                        "optimizing code for training case %d; solution "
                        "may not be optimal" % iters)
    # Attempt to return the exact same object reference.
    if orig_sol is not None and orig_sol.ndim == 1:
        solution = orig_sol
    # Return a vector with the same rank as the input `signals`.
    elif orig_sol is None and signals_ndim == 1:
        solution = solution.squeeze()
    return solution
