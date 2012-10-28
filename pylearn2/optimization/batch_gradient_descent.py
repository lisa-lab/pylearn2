__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import numpy as np
import warnings
from pylearn2.utils import sharedX
from theano import config
from theano import function
from theano.printing import Print
from theano.printing import min_informative_str

import theano.tensor as T

def norm_sq(s):
    return np.square(s.get_value()).sum()

def scale(s, a):
    s.set_value(s.get_value() * np.cast[config.floatX](a))

class BatchGradientDescent:
    """ A class for minimizing a function via the method of steepest
    descent.
    """

    def __init__(self, objective, params, inputs = None,
            param_constrainers = None, max_iter = -1,
            lr_scalers = None, verbose = False, tol = None,
            init_alpha = None,
            reset_alpha = True, hacky_conjugacy = False,
            reset_conjugate = True, gradients = None,
            gradient_updates = None, line_search_mode = None):
        """ objective: a theano expression to be minimized
                       should be a function of params and,
                       if provided, inputs
            params: A list of theano shared variables.
                    These are the optimization variables
            inputs: (Optional) A list of theano variables
                    to serve as inputs to the graph.
            param_constrainers: (Optional) A list of callables
                    to be called on all updates dictionaries to
                    be applied to params. This is how you implement
                    constrained optimization.
            reset_alpha: If True, reverts to using init_alpha after
                        each call. If False, the final set of alphas
                        is used at the start of the next call to minimize.
            hacky_conjugacy: If True, tries to pick conjugate gradient directions.
                        "Hacky" because the conjugate direction equations are only
                        valid on a quadratic function if the line search for the
                        previous search direction ran to completion, but here we
                        just pick the best of k searched positions.
                        I'm not sure if this matters much, since I don't think
                        nonlinear conjugate gradient is all that justified anyway,
                        but then I don't know much about optimization so someone
                        who does might want to look over this file.
            reset_conjugate:
                    has no effect unless hacky_conjugacy == True
                    if reset_conjugate == True,
                        reverts to direction of steepest descent for the first
                        step in each call to minimize.
                    otherwise, tries to make the new search direction
                    conjugate to the last one (even though the objective function
                    might be totally different on each call to minimize)
            gradients: if None, compute the gradients of obj using T.grad
                    otherwise, a dictionary mapping from params to expressions
                    for their gradients (this allows you to use approximate
                    gradients computed with something other than T.grad)
            gradient_updates: a dictionary of shared variable updates to run
                each time the gradient is computed

            Calling the ``minimize'' method with values for
            for ``inputs'' will update ``params'' to minimize
            ``objective''.
        """

        self.__dict__.update(locals())
        del self.self

        if line_search_mode is None:
            if init_alpha is None:
                init_alpha  = ( .001, .005, .01, .05, .1 )
        else:
            assert line_search_mode == 'exhaustive'
            if init_alpha is None:
                init_alpha = (.5, 1.)

        self.init_alpha = tuple([ float(elem) for elem in init_alpha])

        if inputs is None:
            inputs = []

        if param_constrainers is None:
            param_constrainers = []

        obj = objective

        self.verbose = verbose

        param_to_grad_sym = {}
        param_to_grad_shared = {}
        updates = {}
        if self.gradient_updates is not None:
            updates.update(self.gradient_updates)

        self.params = [ param for param in params ]

        for param in params:
            if self.gradients is not None and param in self.gradients:
                grad = self.gradients[param]
            else:
                grad = T.grad(objective, param, disconnected_inputs='ignore')
            param_to_grad_sym[param] = grad
            grad_shared = sharedX( param.get_value() * 0. )
            param_to_grad_shared[param] = grad_shared
            updates[grad_shared] = grad

        self.param_to_grad_shared = param_to_grad_shared

        if self.verbose:
            print 'batch gradient class compiling gradient function'
        self._compute_grad = function(inputs, updates = updates )
        if self.verbose:
            print 'done'

        if self.verbose:
            print 'batch gradient class compiling objective function'
        self.obj = function(inputs, obj)
        if self.verbose:
            print 'done'

        self.param_to_cache = {}
        alpha = T.scalar(name = 'alpha')
        cache_updates = {}
        goto_updates = {}
        for param in params:
            self.param_to_cache[param] = sharedX(param.get_value(borrow=False))
            cache_updates[self.param_to_cache[param]] = param
            cached = self.param_to_cache[param]
            grad = self.param_to_grad_shared[param]
            if lr_scalers is not None and param in lr_scalers:
                scaled_alpha = alpha * lr_scalers[param]
            else:
                scaled_alpha = alpha
            mul = scaled_alpha * grad
            diff = cached - mul
            goto_updates[param] = diff
        self._cache_values = function([],updates = cache_updates)
        for param_constrainer in param_constrainers:
            param_constrainer(goto_updates)
        self._goto_alpha = function([alpha], updates = goto_updates)

        norm = T.sqrt(sum([T.sqr(elem).sum() for elem in self.param_to_grad_shared.values()]))

        normalize_grad_updates = {}
        for grad_shared in self.param_to_grad_shared.values():
            normalize_grad_updates[grad_shared] = grad_shared / norm
        self._normalize_grad = function([], norm, updates = normalize_grad_updates)

        if self.hacky_conjugacy:
            grad_shared = self.param_to_grad_shared.values()

            grad_to_old_grad = {}
            for elem in grad_shared:
                grad_to_old_grad[elem] = sharedX(elem.get_value())

            self._store_old_grad = function([norm], updates = dict([(grad_to_old_grad[grad], grad * norm)
                for grad in grad_to_old_grad]))

            grad_ordered = list(grad_to_old_grad.keys())
            old_grad_ordered = [ grad_to_old_grad[grad] for grad in grad_ordered]

            def dot_product(x, y):
                return sum([ (x_elem * y_elem).sum() for x_elem, y_elem in zip(x, y) ])

            beta_pr = (dot_product(grad_ordered, grad_ordered) - dot_product(grad_ordered, old_grad_ordered)) / \
                    (1e-7+dot_product(old_grad_ordered, old_grad_ordered))
            assert beta_pr.ndim == 0

            beta = T.maximum(beta_pr, 0.)

            """

            beta_pr is the Polak-Ribiere formula for beta.
            According to wikipedia, the beta to use for NCG is "a matter of heuristics or taste"
            but max(0, beta_pr) is "a popular choice... which provides direction reset automatically."
            (ie, it is meant to revert to steepest descent when you have traveled far enough that
            the objective function is behaving non-quadratically enough that the conjugate gradient
            formulas aren't working anymore)

            http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method

            """

            self._make_conjugate = function([], updates = dict([ (grad, grad + beta * grad_to_old_grad[grad]) for
                grad in grad_to_old_grad]))

        if tol is None:
            if objective.dtype == "float32":
                self.tol = 1e-6
            else:
                self.tol = 3e-7
        else:
            self.tol = tol

    """
    def _normalize_grad(self):

        n = sum([norm_sq(elem) for elem in self.param_to_grad_shared.values()])
        n = np.sqrt(n)

        for grad_shared in self.param_to_grad_shared.values():
            scale(grad_shared, 1./n)

        return n
    """

    def minimize(self, * inputs ):

        if self.verbose:
            print 'minimizing'
        alpha_list = list( self.init_alpha )

        orig_obj = self.obj(*inputs)

        if self.verbose:
            print orig_obj


        iters = 0

        # A bit of a hack here: we multiply by norm
        # when calling store_old_grad below. This is mostly
        # so we store the non-normalized version of the gradient,
        # but we can also exploit it to either clear the old grad
        # on the first iteration by setting norm = 0 initially.
        # This makes the algorithm reset to steepest descent on
        # each call to minimize. Or we can set the norm to 1 to
        # save the previous gradient, so we can try to maintain
        # conjugacy across several calls to minimize.
        # If self.hacky_conjugacy is False none of this matters
        # since store_old_grad is never called anyway.
        if self.reset_conjugate:
            norm = 0.
        else:
            norm = 1.

        while iters != self.max_iter:
            iters += 1
            self._cache_values()
            if self.hacky_conjugacy:
                self._store_old_grad(norm)
            self._compute_grad(*inputs)
            if self.hacky_conjugacy:
                self._make_conjugate()
            norm = self._normalize_grad()

            if self.line_search_mode is None:
                best_obj, best_alpha, best_alpha_ind = self.obj( * inputs), 0., -1
                prev_best_obj = best_obj

                for ind, alpha in enumerate(alpha_list):
                    self._goto_alpha(alpha)
                    obj = self.obj(*inputs)
                    if self.verbose:
                        print '\t',alpha,obj

                    #Use <= rather than = so if there are ties
                    #the bigger step size wins
                    if obj <= best_obj:
                        best_obj = obj
                        best_alpha = alpha
                        best_alpha_ind = ind
                    #end if obj
                #end for ind, alpha


                if self.verbose:
                    print best_obj

                assert not np.isnan(best_obj)
                assert best_obj <= prev_best_obj
                self._goto_alpha(best_alpha)

                #if best_obj == prev_best_obj and alpha_list[0] < 1e-5:
                #    break
                if best_alpha_ind < 1 and alpha_list[0] > self.tol:
                    alpha_list = [ alpha / 3. for alpha in alpha_list ]
                    if self.verbose:
                        print 'shrinking the step size'
                elif best_alpha_ind > len(alpha_list) -2:
                    alpha_list = [ alpha * 2. for alpha in alpha_list ]
                    if self.verbose:
                        print 'growing the step size'
                elif best_alpha_ind == -1 and alpha_list[0] <= self.tol:
                    if alpha_list[-1] > 1:
                        if self.verbose:
                            print 'converged'
                        break
                    if self.verbose:
                        print 'expanding the range of step sizes'
                    for i in xrange(len(alpha_list)):
                        for j in xrange(i,len(alpha_list)):
                            alpha_list[j] *= 1.5
                        #end for j
                    #end for i
                else:
                    # if a step succeeded and didn't result in growing or shrinking
                    # the step size then we can probably benefit from more fine-grained
                    # exploration of the middle ranges of step size
                    # (this is especially necessary if we've executed the
                    # 'expanding the range of step sizes' case multiple times)
                    a = np.asarray(alpha_list)
                    s = a[1:]/a[:-1]
                    max_gap = 5.
                    if s.max() > max_gap:
                        weight = .99
                        if self.verbose:
                            print 'shrinking the range of step sizes'
                        alpha_list = [ (alpha ** weight) * (best_alpha ** (1.-weight)) for alpha in alpha_list ]
                        assert all([second > first for first, second in zip(alpha_list[:-1], alpha_list[1:])])
                        # y^(weight) best^(1-weight) / x^(weight) best^(1-weight) = (y/x)^weight
                        # so this shrinks the ratio between each successive pair of alphas by raising it to weight
                        # weight = .99 -> a gap of 5 is shrunk to 4.92


                #end check on alpha_ind
            else:
                assert self.line_search_mode == 'exhaustive'

                # In exhaustive mode, we search until we get very little
                # improvement (or have tried over ten points)
                # and we dynamically pick the search points to try to
                # maximize the improvement.
                # The points we pick are kind of dumb; it's just a binary
                # search. We could probably do better by fitting a function
                # and jumping to its local minima at each step

                print 'Exhaustive line search'


                results = [ (0., self.obj(*inputs) ) ]
                for alpha in alpha_list:
                    assert alpha > results[-1][0]
                    self._goto_alpha(alpha)
                    results.append( (alpha, self.obj(*inputs) ) )
                for alpha, obj in results:
                    print '\t',alpha,obj

                print '\t-------'

                prev_improvement = 0.
                while True:
                    alpha_list = [alpha for alpha, obj in results]
                    obj = [ obj for alpha, obj in results]
                    mn = min(obj)
                    idx = obj.index(mn)

                    def do_point(x):
                        self._goto_alpha(x)
                        res = self.obj(*inputs)
                        print '\t',x,res
                        for i in xrange(len(results)):
                            elem = results[i]
                            ex = elem[0]
                            assert x != ex
                            if x > ex:
                                if i + 1 == len(results) or x < results[i+1][0]:
                                    results.insert(i+1, (x, res))
                                    return mn - res
                        assert False # should be unreached

                    if idx == 0:
                        x = (alpha_list[0] + alpha_list[1]) / 2.
                    elif idx == len(alpha_list) - 1:
                        x = 2 * alpha_list[-1]
                    else:
                        if obj[idx+1] < obj[idx-1]:
                            x = (alpha_list[idx] + alpha_list[idx+1])/2.
                        else:
                            x = (alpha_list[idx] + alpha_list[idx-1])/2.

                    improvement = do_point(x)

                    if improvement > 0 and improvement < .01 * prev_improvement or len(obj) > 10:
                        break
                    prev_improvement = improvement

                alpha_list = [alpha for alpha, obj in results]
                obj = [ obj for alpha, obj in results]
                mn = min(obj)
                idx = obj.index(mn)
                x = alpha_list[idx]
                self._goto_alpha(x)
                print 'final step size: ',x

                alpha_list = [ x/2., x ]
                best_obj = mn

        #end while

        if not self.reset_alpha:
            self.init_alpha = alpha_list

        if norm > 1e-2:
            warnings.warn(str(norm)+" seems pretty big for a gradient at convergence...")

        return best_obj
