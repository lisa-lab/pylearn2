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
            param_constrainers = None, max_iter = -1):
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

            Calling the ``minimize'' method with values for
            for ``inputs'' will update ``params'' to minimize
            ``objective''.
        """

        self.max_iter = max_iter

        if inputs is None:
            inputs = []

        if param_constrainers is None:
            param_constrainers = []

        obj = objective

        self.verbose = False

        param_to_grad_sym = {}
        param_to_grad_shared = {}
        updates = {}

        self.params = [ param for param in params ]

        for param in params:
            grad = T.grad(objective, param)
            param_to_grad_sym[param] = grad
            grad_shared = sharedX( param.get_value() )
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
            mul = alpha * grad
            diff = cached - mul
            goto_updates[param] = diff
        self._cache_values = function([],updates = cache_updates)
        for param_constrainer in param_constrainers:
            param_constrainer(goto_updates)
        self._goto_alpha = function([alpha], updates = goto_updates)

    def _normalize_grad(self):

        n = sum([norm_sq(elem) for elem in self.param_to_grad_shared.values()])
        n = np.sqrt(n)

        for grad_shared in self.param_to_grad_shared.values():
            scale(grad_shared, 1./n)

        return n

    def minimize(self, * inputs ):

        alpha_list = [ .001, .005, .01, .05, .1 ]

        orig_obj = self.obj(*inputs)

        if self.verbose:
            print orig_obj


        iters = 0
        while iters != self.max_iter:
            iters += 1
            best_obj, best_alpha, best_alpha_ind = self.obj( * inputs), 0., -1
            self._cache_values()
            self._compute_grad(*inputs)
            norm = self._normalize_grad()

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

            if best_obj == prev_best_obj:
                break

            if best_alpha_ind < 1 and alpha_list[0] > 3e-7:
                alpha_list = [ alpha / 3. for alpha in alpha_list ]
            elif best_alpha_ind > len(alpha_list) -2:
                alpha_list = [ alpha * 2. for alpha in alpha_list ]
            elif best_alpha_ind == -1 and alpha_list[0] <= 3e-7:
                if alpha_list[-1] > 1:
                    break
                for i in xrange(len(alpha_list)):
                    for j in xrange(i,len(alpha_list)):
                        alpha_list[j] *= 1.5
                    #end for j
                #end for i
            #end check on alpha_ind
        #end while



        if norm > 1e-2:
            warnings.warn(str(norm)+" seems pretty big for a gradient at convergence...")

        return best_obj
