"""
Feature sign search algorithm for solving for L1-penalized sparse codes,
almost a literal translation from Honglak Lee's MATLAB implementation.
"""
import scipy.linalg as slinalg
import numpy as np

def l1ls_featuresign(A, Y, gamma, Xinit=None, maxiter=1000):
    """
    Use the feature sign search algorithm of Lee et al (2006) to solve
    for L1-penalized sparse codes, i.e. argmin_x (y - Ax)^2 + gamma |x|_1.
    """
    numdim, numpoints = Y.shape
    _, numbases = A.shape
    assert numdim == A.shape[0], "A and Y must have same first dimension"
    if Xinit is not None:
        assert numbases == Xinit.shape[1], ("Xinit is wrong shape for given "
                                            + "basis matrix A")
    Xout = np.zeros((numbases, numpoints))
    AtA = np.dot(A.T, A)
    try:
        AtY = np.dot(A.T, Y)
    except ValueError, e:
        raise ValueError('Got exception '+str(e)+' while trying to do np.dot(A.T,Y) with A having shape '+str(A.shape)+
                'and Y having shape '+str(Y.shape))
    rankA = np.amin(A.shape) - 10 # Why -10?
    for i in range(numpoints):
        if Xinit is not None:
            idx1, = np.where(Xinit[:, i] != 0)
            maxn = min(len(idx1), rankA)
            xinit = np.zeros(Xinit.shape[1])
            xinit[idx1[:maxn]] = Xinit[idx1[:maxn], i]
            Xout[:, i], fobj = ls_featuresign_sub(A, Y[:, i], AtA, AtY[:, i],
                                                  gamma, xinit,
                                                  maxiter=maxiter)
        else:
            Xout[:, i], fobj = ls_featuresign_sub(A, Y[:, i], AtA, AtY[:, i],
                                                  gamma, maxiter=maxiter)

    return Xout

def ls_featuresign_sub(A, y, AtA, Aty, gamma, xinit=None, maxiter=1000):
    """
    Solve for a single sparse code given the Gram matrix ``AtA`` and the
    projection ``Aty``, and L1 penalty coefficient ``gamma``.
    """
    L, M = A.shape
    rankA = np.min(A.shape) - 10
    usexinit = False
    if xinit is None or len(xinit) == 0:
        x, theta, act = np.zeros((3, M))
        allowZero = False
    else:
        x = xinit
        theta = np.sign(x)
        act = np.abs(theta)
        usexinit = True
        allowZero = True
    fobj = 0
    optimality1 = False
    for iteration in xrange(maxiter):
        act_indx0, = np.where(act == 0)
        grad = np.dot(AtA, x) - Aty
        theta = np.sign(x)
        optimality0 = False
        aa = np.abs(grad[act_indx0])
        if len(aa) > 0:
            indx = np.argmax(aa)
        else:
            indx = []
        mx = aa[indx]
        if len(aa) != 0 and mx >= gamma and (iteration > 0 or not usexinit):
            act[act_indx0[indx]] = 1
            theta[act_indx0[indx]] = -np.sign(grad[act_indx0[indx]])
            usexinit = False
        else:
            optimality0 = True
            if optimality1:
                break
        act_indx1, = np.where(act == 1)
        if len(act_indx1) > rankA:
            print 'Warning: sparsity penalty too small'
        if len(act_indx1) == 0:
            if allowZero:
                allowZero = False
                continue
            else:
                return x, fobj
        k = 0
        while True:
            k = k + 1
            if k > maxiter:
                print ("Warning: Maximum number of iterations reached. "
                       + "Solution may not be optimal")
                return x, fobj
            if len(act_indx1) == 0:
                if allowZero:
                    allowZero = False
                    break
                else:
                    return x, fobj

            x, theta, act, act_indx1, optimality1, lsearch, fobj = \
                    compute_fs_step(x, A, y, AtA, Aty, theta, act, act_indx1,
                                    gamma)
            if optimality1:
                break
            if lsearch > 0:
                continue
    if iteration >= maxiter:
        print ("Warning: Maximum number of iterations reached. "
               + "Solution may not be optimal")
    fobj = fobj_featuresign(x, A, y, AtA, Aty, gamma)
    return x, fobj

def compute_fs_step(x, A, y, AtA, Aty, theta, act, act_indx1, gamma):
    """
    Solve the unconstrained quadratic program implied by a certain choice of
    coefficient signs ``theta``. Then do a line search between x and x_new
    and evaluate the cost function at all points where any coefficient changes
    sign, and choose the one with the lowest objective function value.
    """
    theta = theta.copy()
    act = act.copy()
    act_indx1 = act_indx1.copy()
    x2 = x[act_indx1]
    AtA2 = AtA[np.ix_(act_indx1, act_indx1)]
    theta2 = theta[act_indx1]
    x_new = slinalg.solve(AtA2, (Aty[act_indx1] - gamma * theta2),
                          sym_pos=True)
    optimality1 = False
    if np.all(np.sign(x_new) == np.sign(x2)):
        # If the signs haven't changed at all between x2 and x_new,
        # optimality condition a) in the paper is satisfied.
        optimality1 = True
        x[act_indx1] = x_new
        fobj = 0.
        lsearch = 1.
        return x, theta, act, act_indx1, optimality1, lsearch, fobj
    progress = (0. - x2) / (x_new - x2)
    lsearch = 0.
    a = 0.5 * (np.dot(A[:, act_indx1], (x_new - x2))**2).sum(axis=0)
    b = (np.dot(np.dot(x2.T, AtA2), x_new - x2)
         - np.dot((x_new - x2).T, Aty[act_indx1]))
    fobj_lsearch = gamma * np.abs(x2).sum(axis=0)
    prog1 = np.concatenate((progress, [1]))
    ix_lsearch = np.argsort(prog1)
    sort_lsearch = prog1[ix_lsearch]
    remove_idx = []
    for i in range(len(sort_lsearch)):
        t = sort_lsearch[i]
        if t <= 0 or t > 1:
            continue
        s_temp = x2 + (x_new - x2) * t
        # Matrix multiplies?
        fobj_temp = a * t**2 + b * t + gamma * np.abs(s_temp).sum(axis=0)
        if fobj_temp < fobj_lsearch:
            fobj_lsearch = fobj_temp
            lsearch = t
            if t < 1:
                remove_idx.append(ix_lsearch[i])
        elif fobj_temp > fobj_lsearch:
            break
        else:
            if (x2 == 0).sum() == 0:
                lsearch = t
                fobj_lsearch = fobj_temp
                if t < 1:
                    remove_idx.append(ix_lsearch[i])
    if lsearch > 0:
        x_new = x2 + (x_new - x2) * lsearch
        x[act_indx1] = x_new
        theta[act_indx1] = np.sign(x_new)
    if lsearch < 1 and lsearch > 0:
        remove_idx, = np.where(np.abs(x[act_indx1]) < np.finfo('float64').eps)
        x[act_indx1[remove_idx]] = 0
        theta[act_indx1[remove_idx]] = 0
        act[act_indx1[remove_idx]] = 0
        remaining_idx = np.setdiff1d(np.arange(len(act_indx1)), remove_idx)
        act_indx1 = act_indx1[np.sort(remaining_idx)]
    # commented in matlab code:
    fobj_new = 0 # fobj_featuresign(x, A, y, AtA, Aty, gamma)
    fobj = fobj_new
    return x, theta, act, act_indx1, optimality1, lsearch, fobj

def fobj_featuresign(x, A, y, AtA, Aty, gamma, getg=True):
    """
    Return the quadratic objective function and (optionally) the
    gradient.
    """
    f = 0.5 * np.linalg.norm(y - np.dot(A, x))**2
    f = f + gamma * np.linalg.norm(x, 1)
    if getg:
        g = np.dot(AtA, x) - Aty
        g = g + gamma * np.sign(x)
    else:
        g = None
    return f, g

