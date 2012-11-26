"""
Note: this code is inspired from the following matlab source :
    http://www.stanford.edu/group/SOL/software/minres.html
"""

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
import numpy
from pylearn2.utils import constantX
from pylearn2.expr.basic import multiple_switch, symGivens2, \
        sqrt_inner_product, inner_product

# Messages that matches the flag value returned by the method
messages = [
    ' beta1 = 0.  The exact solution is  x = 0.                    ',  # 0
    ' A solution to (poss. singular) Ax = b found, given rtol.     ',  # 1
    ' A least-squares solution was found, given rtol.              ',  # 2
    ' A solution to (poss. singular) Ax = b found, given eps.      ',  # 3
    ' A least-squares solution was found, given eps.               ',  # 4
    ' x has converged to an eigenvector.                           ',  # 5
    ' xnorm has exceeded maxxnorm.                                 ',  # 6
    ' Acond has exceeded Acondlim.                                 ',  # 7
    ' The iteration limit was reached.                             ',  # 8
    ' A least-squares solution for singular LS problem, given eps. ',  # 9
    ' A least-squares solution for singular LS problem, given rtol.',  # 10
    ' A null vector obtained, given rtol.                          ',  # 11
    ' Numbers are too small to continue computation                ']  # 12


def minres(compute_Av,
           bs,
           rtol=constantX(1e-6),
           maxit=20,
           Ms=None,
           shift=constantX(0.),
           maxxnorm=constantX(1e15),
           Acondlim=constantX(1e16),
           profile=0):
    """
     minres attempts to find the minimum-length and minimum-residual-norm
     solution x to the system of linear equations A*x = b or
     least squares problem min||Ax-b||.  The n-by-n coefficient matrix A
     must be symmetric (but need not be positive definite or invertible).
     The right-hand-side column vector b must have length n.

     Parameters:

        compute_Av: callable returing the symbolic expression for
            `Av` (the product of matrix A with some vector v).
            `v` should be a list of tensors, whre the vector v means
            the vector obtain by concatenating and flattening all tensors in
            v
        bs: list of Theano expressions. We are looking to compute
            `A^-1\dot bs`.
        rtol: Optional, real, specifies the tolerance of the method.
            Default is 1e-6
        maxit: Optional, positive integer, specifies the maximum number
            of iterations. Default is 20
        Ms: List of theano expression of same shape as `bs`. The
            method uses these to precondition with diag(Ms)
        shift: Optional, scalar, real or complex.  Default is 0.
                   Effectively solve the system (A - shift I) * x = b.
        maxxnorm   real positive, maximum bound on NORM(x). Default is 1e14.
        Acondlim   real positive, maximum bound on COND(A). Default is 1e15.
        show       boolean, 0 to suppress outputs, 1 to show iterations.
                   Default is 0.

     OUTPUTS:
        x       list of Theano tensor representing the solution
        flag    theano int scalar - convergence flag
                0 beta1 = 0.  The exact solution is  x = 0.
                1 A solution to (poss. singular) Ax = b found, given rtol.
                2 Pseudoinverse solution for singular LS problem, given rtol.
                3 A solution to (poss. singular) Ax = b found, given eps.
                4 Pseudoinverse solution for singular LS problem, given eps.
                5 x has converged to an eigenvector.
                6 xnorm has exceeded maxxnorm.
                7 Acond has exceeded Acondlim.
                8 The iteration limit was reached.
                9/10 It is a least squares problem but no converged
                 solution yet.
        iter    integer, iteration number at which x was computed:
                0 <= iter <= maxit.
        relres  real positive, the relative residual is defined as
                     NORM(b-A*x)/(NORM(A) * NORM(x) + NORM(b)),
                computed recurrently here.  If flag is 1 or 3,  relres <= TOL.
        relAres real positive, the relative-NORM(Ar) := NORM(Ar) / NORM(A) ---
                computed recurrently here. If flag is 2 or 4, relAres <= TOL.
        Anorm   real positive, estimate of matrix 2-norm of A.
        Acond   real positive, estimate of condition number of A with
                respect to 2-norm.
        xnorm   non-negative positive, recurrently computed NORM(x)
        Axnorm  non-negative positive, recurrently computed NORM(A * x).

     REFERENCES:
        Sou-Cheng Choi's PhD Dissertation, Stanford University, 2006.
             http://www.stanford.edu/group/SOL/software.html

    """

    if not isinstance(bs, (tuple, list)):
        bs = [bs]
        return_as_list = False
    else:
        bs = list(bs)
        return_as_list = True

    eps = constantX(1e-23)

    # Initialise
    flag = theano.shared(constantX(0.))
    beta1 = sqrt_inner_product(bs)

    #------------------------------------------------------------------
    # Set up p and v for the first Lanczos vector v1.
    # p  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.
    #------------------------------------------------------------------
    r3s = [b for b in bs]
    r2s = [b for b in bs]
    r1s = [b for b in bs]
    if Ms is not None:
        r3s = [b / m for b, m in zip(bs, Ms)]
        beta1 = sqrt_inner_product(r3s, bs)
    #------------------------------------------------------------------
    ## Initialize other quantities.
    # Note that Anorm has been initialized by IsOpSym6.
    # ------------------------------------------------------------------
    bnorm = beta1
    n_params = len(bs)

    def loop(niter,
             beta,
             betan,
             phi,
             Acond,
             cs,
             dbarn,
             eplnn,
             rnorm,
             sn,
             Tnorm,
             rnorml,
             xnorm,
             Dnorm,
             gamma,
             pnorm,
             gammal,
             Axnorm,
             relrnorm,
             relArnorml,
             Anorm,
             flag,
             *args):
        #-----------------------------------------------------------------
        ## Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
        # The general iteration is similar to the case k = 1 with v0 = 0:
        #
        #   p1      = Operator * v1  -  beta1 * v0,
        #   alpha1  = v1'p1,
        #   q2      = p2  -  alpha1 * v1,
        #   beta2^2 = q2'q2,
        #   v2      = (1/beta2) q2.
        #
        # Again, p = betak P vk,  where  P = C**(-1).
        # .... more description needed.
        #-----------------------------------------------------------------
        xs = args[0 * n_params: 1 * n_params]
        r1s = args[1 * n_params: 2 * n_params]
        r2s = args[2 * n_params: 3 * n_params]
        r3s = args[3 * n_params: 4 * n_params]
        dls = args[4 * n_params: 5 * n_params]
        ds = args[5 * n_params: 6 * n_params]
        betal = beta
        beta = betan
        vs = [r3 / beta for r3 in r3s]
        r3s, upds = compute_Av(*vs)

        r3s = [r3 - shift * v for r3, v in zip(r3s, vs)]
        r3s = [TT.switch(TT.ge(niter, constantX(1.)),
                         r3 - (beta / betal) * r1,
                         r3) for r3, r1 in zip(r3s, r1s)]

        alpha = inner_product(r3s, vs)
        r3s = [r3 - (alpha / beta) * r2 for r3, r2 in zip(r3s, r2s)]
        r1s = [r2 for r2 in r2s]
        r2s = [r3 for r3 in r3s]
        if Ms is not None:
            r3s = [r3 / M for r3, M in zip(r3s, Ms)]
            betan = sqrt_inner_product(r2s, r3s)
        else:
            betan = sqrt_inner_product(r3s)
        pnorml = pnorm
        pnorm = TT.switch(TT.eq(niter, constantX(0.)),
                          TT.sqrt(TT.sqr(alpha) + TT.sqr(betan)),
                          TT.sqrt(TT.sqr(alpha) + TT.sqr(betan) +
                                  TT.sqr(beta)))

        #-----------------------------------------------------------------
        ## Apply previous rotation Qk-1 to get
        #   [dlta_k epln_{k+1}] = [cs  sn][dbar_k    0      ]
        #   [gbar_k  dbar_{k+1} ]   [sn -cs][alpha_k beta_{k+1}].
        #-----------------------------------------------------------------
        dbar = dbarn
        epln = eplnn
        dlta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha

        eplnn = sn * betan
        dbarn = -cs * betan

        ## Compute the current plane rotation Qk
        gammal2 = gammal
        gammal = gamma
        cs, sn, gamma = symGivens2(gbar, betan)
        tau = cs * phi
        phi = sn * phi
        Axnorm = TT.sqrt(TT.sqr(Axnorm) + TT.sqr(tau))
        # Update d

        dl2s = [dl for dl in dls]
        dls = [d for d in ds]
        ds = [TT.switch(TT.neq(gamma, constantX(0.)),
                        (v - epln * dl2 - dlta * dl) / gamma,
                        v)
              for v, dl2, dl in zip(vs, dl2s, dls)]
        d_norm = TT.switch(TT.neq(gamma, constantX(0.)),
                           sqrt_inner_product(ds),
                           constantX(numpy.inf))

        # Update x except if it will become too big
        xnorml = xnorm
        dl2s = [x for x in xs]
        xs = [x + tau * d for x, d in zip(xs, ds)]

        xnorm = sqrt_inner_product(xs)
        xs = [TT.switch(TT.ge(xnorm, maxxnorm),
                        dl2, x)
              for dl2, x in zip(dl2s, xs)]

        flag = TT.switch(TT.ge(xnorm, maxxnorm),
                         constantX(6.), flag)
        # Estimate various norms
        rnorml = rnorm  # ||r_{k-1}||
        Anorml = Anorm
        Acondl = Acond
        relrnorml = relrnorm
        flag_no_6 = TT.neq(flag, constantX(6.))
        Dnorm = TT.switch(flag_no_6,
                          TT.sqrt(TT.sqr(Dnorm) + TT.sqr(d_norm)),
                          Dnorm)
        xnorm = TT.switch(flag_no_6, sqrt_inner_product(xs), xnorm)
        rnorm = TT.switch(flag_no_6, phi, rnorm)
        relrnorm = TT.switch(flag_no_6,
                             rnorm / (Anorm * xnorm + bnorm),
                             relrnorm)
        Tnorm = TT.switch(flag_no_6,
                          TT.switch(TT.eq(niter, constantX(0.)),
                                    TT.sqrt(TT.sqr(alpha) + TT.sqr(betan)),
                                    TT.sqrt(TT.sqr(Tnorm) +
                                            TT.sqr(beta) +
                                            TT.sqr(alpha) +
                                            TT.sqr(betan))),
                          Tnorm)
        Anorm = TT.maximum(Anorm, pnorm)
        Acond = Anorm * Dnorm
        rootl = TT.sqrt(TT.sqr(gbar) + TT.sqr(dbarn))
        Anorml = rnorml * rootl
        relArnorml = rootl / Anorm

        #---------------------------------------------------------------
        # See if any of the stopping criteria are satisfied.
        # In rare cases, flag is already -1 from above (Abar = const*I).
        #---------------------------------------------------------------
        epsx = Anorm * xnorm * eps
        epsr = Anorm * xnorm * rtol
        #Test for singular Hk (hence singular A)
        # or x is already an LS solution (so again A must be singular).
        t1 = constantX(1) + relrnorm
        t2 = constantX(1) + relArnorml

        flag = TT.switch(
            TT.bitwise_or(TT.eq(flag, constantX(0)),
                          TT.eq(flag, constantX(6))),
            multiple_switch(TT.le(t1, constantX(1)),
                            constantX(3),
                            TT.le(t2, constantX(1)),
                            constantX(4),
                            TT.le(relrnorm, rtol),
                            constantX(1),
                            TT.le(Anorm, constantX(1e-20)),
                            constantX(12),
                            TT.le(relArnorml, rtol),
                            constantX(10),
                            TT.ge(epsx, beta1),
                            constantX(5),
                            TT.ge(xnorm, maxxnorm),
                            constantX(6),
                            TT.ge(niter, TT.cast(maxit,
                                                 theano.config.floatX)),
                            constantX(8),
                            flag),
            flag)

        flag = TT.switch(TT.lt(Axnorm, rtol * Anorm * xnorm),
                         constantX(11.),
                         flag)
        return [niter + constantX(1.),
                beta,
                betan,
                phi,
                Acond,
                cs,
                dbarn,
                eplnn,
                rnorm,
                sn,
                Tnorm,
                rnorml,
                xnorm,
                Dnorm,
                gamma,
                pnorm,
                gammal,
                Axnorm,
                relrnorm,
                relArnorml,
                Anorm,
                flag] + xs + r1s + r2s + r3s + dls + ds, upds, \
                theano.scan_module.scan_utils.until(TT.neq(flag, 0))

    states = []
    # 0 niter
    states.append(constantX([0]))
    # 1 beta
    states.append(constantX([0]))
    # 2 betan
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 3 phi
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 4 Acond
    states.append(constantX([1]))
    # 5 cs
    states.append(constantX([-1]))
    # 6 dbarn
    states.append(constantX([0]))
    # 7 eplnn
    states.append(constantX([0]))
    # 8 rnorm
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 9 sn
    states.append(constantX([0]))
    # 10 Tnorm
    states.append(constantX([0]))
    # 11 rnorml
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 12 xnorm
    states.append(constantX([0]))
    # 13 Dnorm
    states.append(constantX([0]))
    # 14 gamma
    states.append(constantX([0]))
    # 15 pnorm
    states.append(constantX([0]))
    # 16 gammal
    states.append(constantX([0]))
    # 17 Axnorm
    states.append(constantX([0]))
    # 18 relrnorm
    states.append(constantX([1]))
    # 19 relArnorml
    states.append(constantX([1]))
    # 20 Anorm
    states.append(constantX([0]))
    # 21 flag
    states.append(constantX([0]))
    xs = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    ds = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    dls = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    r1s = [TT.unbroadcast(TT.shape_padleft(r1), 0) for r1 in r1s]
    r2s = [TT.unbroadcast(TT.shape_padleft(r2), 0) for r2 in r2s]
    r3s = [TT.unbroadcast(TT.shape_padleft(r3), 0) for r3 in r3s]

    rvals, loc_updates = scan(
        loop,
        states=states + xs + r1s + r2s + r3s + dls + ds,
        n_steps=maxit + numpy.int32(1),
        name='minres',
        profile=profile,
        mode=theano.Mode(linker='cvm'))
    assert isinstance(loc_updates, dict) and 'Ordered' in str(type(loc_updates))

    niters = TT.cast(rvals[0][0], 'int32')
    flag = TT.cast(rvals[21][0], 'int32')
    relres = rvals[18][0]
    relAres = rvals[19][0]
    Anorm = rvals[20][0]
    Acond = rvals[4][0]
    xnorm = rvals[12][0]
    Axnorm = rvals[17][0]
    sol = [x[0] for x in rvals[22: 22 + n_params]]
    return (sol,
            flag,
            niters,
            relres,
            relAres,
            Anorm,
            Acond,
            xnorm,
            Axnorm,
            loc_updates)


