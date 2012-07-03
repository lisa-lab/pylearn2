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

    def __init__(self, objective, params, inputs = None):
        """ objective: a theano expression to be minimized
                       should be a function of params and,
                       if provided, inputs
            params: A list of theano shared variables.
                    These are the optimization variables
            inputs: (Optional) A list of theano variables
                    to serve as inputs to the graph.

            Calling the ``minimize'' method with values for
            for ``inputs'' will update ``params'' to minimize
            ``objective''.
        """
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

    def _cache_values(self):

        self.cache = [ param.get_value() for param in self.params ]

    def _goto_alpha(self, alpha):

        for param, cached_value, grad in zip(self.params, self.cache):
            assert not np.any(np.isnan(cached_value))
            grad_shared = self.param_to_grad_shared[param]
            grad = self.grad_H.get_value()
            assert not np.any(np.isnan(grad))
            assert not np.any(np.isinf(grad))
            mul = alpha * grad
            assert not np.any(np.isnan(mul))
            diff = cached_value - mul
            param.set_value(diff)

    def _normalize_grad(self):

        n = sum([norm_sq(elem) for elem in self.param_to_grad_shared.keys()])
        n = np.sqrt(n)

        for param in self.params:
            scale(param, 1./n)

    def minimize(self, * inputs ):

        alpha_list = [ .001, .005, .01, .05, .1 ]

        orig_obj = self.obj(*inputs)

        if self.verbose:
            print orig_obj

        while True:

            best_obj, best_alpha, best_alpha_ind = self.obj( * inputs), 0., -1
            assert best_obj <= orig_obj
            self.cache_values()
            self._compute_grad(*inputs)
            self._normalize_grad()

            prev_best_obj = best_obj

            for ind, alpha in enumerate(alpha_list):
                self._goto_alpha(alpha)
                obj = self.obj(*inputs)
                if self.verbose:
                    print '\t',alpha,obj

                if obj < best_obj:
                    best_obj = obj
                    best_alpha = alpha
                    best_alpha_ind = ind

            if self.verbose:
                print best_obj

            assert not np.isnan(best_obj)
            assert best_obj <= prev_best_obj
            self._goto_alpha(best_alpha)

            if best_alpha_ind < 1 and alpha_list[0] > 3e-7:
                alpha_list = [ alpha / 3. for alpha in alpha_list ]
            elif best_alpha_ind > len(alpha_list) -2:
                alpha_list = [ alpha * 2. for alpha in alpha_list ]
            elif best_alpha_ind == -1 and alpha_list[0] <= 3e-7:
                break

        return best_obj
