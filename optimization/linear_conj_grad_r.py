from theano import function, shared
import theano.tensor as T
import scipy.linalg
from theano.printing import Print
import numpy as N

def linear_conj_grad_r( f, x, tol = 1e-3, max_iters = 1000):
    """
        Minimizes a POSITIVE DEFINITE quadratic function
        via linear conjugate gradient using the R operator
        to avoid explicitly representing the Hessian

        If you have several variables, this is cheaper than
        Newton's method, which would need to invert the
        Hessian. It is also cheaper than standard linear
        conjugate gradient, which works with an explicit
        representation of the Hessian. It is also cheaper
        than nonlinear conjugate gradient which does a
        line search by repeatedly evaluating f.

        Parameters:

            f: a theano expression which is quadratic with
               POSITIVE DEFINITE hessian in x
            x: a list of theano shared variables that influence f

            tol: minimization halts when the norm of the gradient
                is smaller than tol


        Return:
            None

            x will be modified so that its values minimize f


        Reference:
            http://komarix.org/ac/papers/thesis/thesis_html/node11.html

            (This reference describes linear CG but not converting it to use
            the R operator instead of an explicit representation of the Hessian)

    """



    beta = shared(0.0, name='beta')

    first = True

    #define functions

    residuals = [ shared( p.get_value(borrow=False) ) for p in x ]
    old_directions = [ shared( p.get_value(borrow=False) ) for p in x]
    for i in xrange(len(residuals)):
        if x[i].name is None:
            residuals[i].name = 'residuals[%d]' % i
            old_directions[i].name = 'old_directions[%d]' % i
        else:
            residuals[i].name = x[i].name+' residual'
            old_directions[i].name = x[i].name +' old direction'
        #
    #

    compute_residuals = function([],updates =
            [
             (residual, -T.grad(f,x_i))
             for residual, x_i in zip(residuals,x)
            ])

    residual_norm_squared = shared(0.0, name = 'residual_norm_squared')
    old_residual_norm_squared = shared(0.0, name = 'old_residual_norm_squared')

    residual_norm_squared_var = sum([ T.sum(T.sqr(residual)) for residual in residuals ])
    residual_norm_squared_var.name = 'residual_norm_squared_var'

    residual_norm_var = T.sqrt(residual_norm_squared_var)
    residual_norm_var.name = 'residual_norm_var'

    residual_norm = function([], residual_norm_var,
            updates = [ (residual_norm_squared, residual_norm_squared_var) ],
            name='residual_norm')

    beta_var = residual_norm_squared / old_residual_norm_squared

    compute_beta = function([], updates = [ (beta, beta_var) ] )


    iteration_updates = []

    directions = [ residual + beta * old_direction for residual, old_direction
                    in zip(residuals, old_directions) ]


    A_i_dot_r = T.R_op(T.grad(f,x),x,residuals)

    #TODO-- sum of elemwise product can be optimized to dot in many cases
    #       is this optimization actually getting applied? if not, make it so
    r_A_r = sum( [ T.sum(r_i * A_ri) for r_i, A_ri in zip(residuals,A_i_dot_r) ] )


    alpha =  residual_norm_squared / r_A_r

    iteration_updates.append( (old_residual_norm_squared, residual_norm_squared) )

    for direction, old_direction in zip(directions,old_directions):
        iteration_updates.append((old_direction,direction))
    #

    for p, d in zip(x,directions):
        iteration_updates.append((p,p+alpha*d))
    #

    iterate = function([], updates =  iteration_updates)


    #main loop

    compute_residuals()



    iters = 0

    while residual_norm() > tol and iters < max_iters:
        print residual_norm()

        if not first:
            compute_beta()
        #

        iterate()

        iters += 1

        compute_residuals()
    #

    print 'finished with a gradient norm of %d after %d iters' % (residual_norm(), iters)
#


def linear_conj_grad_r_hack( f, x, sequential_vars, sequences, tol = 1e-3, max_iters = 1000):
    """
        This is a temporary hack for minimizing a function
        that is additive over a minibatch.

        Will be obsolete once R_op is implemented for scan
    """

    beta = shared(0.0, name='beta')

    first = True

    #define functions

    residuals = [ shared( p.get_value(borrow=False) ) for p in x ]
    old_directions = [ shared( p.get_value(borrow=False) ) for p in x]
    for i in xrange(len(residuals)):
        if x[i].name is None:
            residuals[i].name = 'residuals[%d]' % i
            old_directions[i].name = 'old_directions[%d]' % i
        else:
            residuals[i].name = x[i].name+' residual'
            old_directions[i].name = x[i].name +' old direction'
        #
    #

    zero_residuals = function([], updates = [ (residual, T.zeros_like(residual) )
                     for residual in residuals] )

    inc_residuals = function(sequential_vars,updates =
            [
             (residual, residual -T.grad(f,x_i))
             for residual, x_i in zip(residuals,x)
            ])


    def compute_residuals():
        zero_residuals()
        for i in xrange(sequences[0].shape[0]):
            inc_residuals( * [ s[i,:] for s in sequences ] )
        #
    #


    residual_norm_squared = shared(0.0, name = 'residual_norm_squared')
    old_residual_norm_squared = shared(0.0, name = 'old_residual_norm_squared')

    residual_norm_squared_var = sum([ T.sum(T.sqr(residual)) for residual in residuals ])
    residual_norm_squared_var.name = 'residual_norm_squared_var'

    residual_norm_var = T.sqrt(residual_norm_squared_var)
    residual_norm_var.name = 'residual_norm_var'

    residual_norm = function([], residual_norm_var,
            updates = [ (residual_norm_squared, residual_norm_squared_var) ],
            name='residual_norm')

    beta_var = residual_norm_squared / old_residual_norm_squared

    compute_beta = function([], updates = [ (beta, beta_var) ] )


    iteration_updates = []

    directions = [ residual + beta * old_direction for residual, old_direction
                    in zip(residuals, old_directions) ]


    #note- this is a list because x is
    jacobian = T.grad(f,x)

    #try:
    A_i_dot_r = T.R_op(jacobian,x,residuals)
    #except Exception, e:
        #print jacobian
        #import IPython
        #embed = IPython.Shell.IPShellEmbed()
        #embed()
     #   raise Exception("couldn't do R operator due to "+str(e))


    #TODO-- sum of elemwise product can be optimized to dot in many cases
    #       is this optimization actually getting applied? if not, make it so
    r_A_r = sum( [ T.sum(r_i * A_ri) for r_i, A_ri in zip(residuals,A_i_dot_r) ] )

    r_A_r_shared = shared(0.0)

    zero_r_A_r = function([],updates=[(r_A_r_shared,0.0)])
    inc_r_A_r = function(sequential_vars,updates=[(r_A_r_shared,r_A_r_shared+r_A_r)])

    def compute_rAr():
        zero_r_A_r()
        for i in xrange(sequences[0].shape[0]):
            inc_r_A_r( * [ s[i,:] for s in sequences ] )
        #
    #


    alpha =  residual_norm_squared / r_A_r_shared

    iteration_updates.append( (old_residual_norm_squared, residual_norm_squared) )

    for direction, old_direction in zip(directions,old_directions):
        iteration_updates.append((old_direction,direction))
    #

    for p, d in zip(x,directions):
        iteration_updates.append((p,p+alpha*d))
    #

    iterate = function([], updates =  iteration_updates)


    #main loop

    compute_residuals()


    iters = 0

    while (residual_norm() > tol) and (iters < max_iters):
        print 'residual norm: '+str(residual_norm())

        if not first:
            compute_beta()
        #

        compute_rAr()

        print 'rAr: '+str(r_A_r_shared.get_value())

        iterate()

        for x_i in x:
            assert not N.any(N.isnan(x_i.get_value(borrow=True)) )
            assert not N.any(N.isinf(x_i.get_value(borrow=True)))

        iters += 1

        compute_residuals()

    #

    print 'finished with a gradient norm of %d after %d iters' % (residual_norm(), iters)
#




if __name__ == '__main__':
    import numpy as N

    rng = N.random.RandomState([1,2,3])

    n = 5

    M = rng.randn(2*n,n)

    M = N.dot(M.T,M)

    b = rng.randn(n)
    c = rng.randn()

    x = shared(rng.randn(n))


    f = 0.5 * T.dot(x,T.dot(M,x)) - T.dot(b,x) + c


    linear_conj_grad_r(f,[x])

    eval_f = function([],f)

    print 'value of f: '+str(eval_f())

    x.set_value(   scipy.linalg.solve(M,b) , borrow = True )

    print 'true minimum value: '+str(eval_f())
