from collections import OrderedDict
import warnings
import theano.tensor as T

class Constraint(object):

    """
        Base class for implementing constraints.
    """
    def __init__(self, axis=None):
        """
            axis: determines which axises to apply the constraints on.
        """
        axis = axis if axis is not None else (0,)
        self.axis = axis

    def constrain_params(self, constraint_val):
        raise NotImplementedError()

class NormConstraint(Constraint):
    """
        A class implementing norm constraints. This class can be used to implement max/min norm
        constraints on the parameters of the model.
    """
    def __init__(self, axis=(0,), dimshuffle_pattern=None):
        """
            axis: determines which axises to apply the constraints on.
            dimshuffle_pattern: determines which dimshuffling pattern to apply the dimshuffling
            pattern on.
        """
        self.dimshuffle_pattern = dimshuffle_pattern
        super(NormConstraint, self).__init__(axis)

    def _clip_norms(self,
                    init_param,
                    max_norm,
                    min_norm,
                    eps):
        """
            init_param: The parameter that we are going to apply the constraint on.
            max_norm: Maximum norm constraint.
            min_norm: Minimum norm constraint.
            eps: Epsilon, a small value to be added to norm for numerical stability to ensure that
            denominator never becomes 0.
        """

        axis = self.axis if self.axis is not None else (0,)
        assert type(axis) is tuple, "Axis you have provided to the norm constraint class, should be a tuple."
        assert max_norm is not None
        min_norm_constr = min_norm if min_norm is not None else 0
        dimshuffle_pattern = self.dimshuffle_pattern

        sqr_param = T.sqr(init_param)
        norm = T.sqrt(T.sum(sqr_param, axis=axis))
        desired_norm = T.clip(norm, min_norm_constr, max_norm)
        desired_norm_ratio = desired_norm / (eps + norm)

        if dimshuffle_pattern is not None:
            desired_norm_ratio = desired_norm_ratio.dimshuffle(dimshuffle_pattern)
        clipped_param = init_param * desired_norm_ratio

        return clipped_param


    def constrain_updates(self,
                          params=None,
                          updates=None,
                          min_norm_constraint=None,
                          max_norm_constraint=None,
                          sort_params=True,
                          eps=1e-7):
        """
        Apply the constraint on the updates of the model.
            params: A dictionary of the name of paramaters that the constraint is going to be applied.
            updates: Updates that we are going to update our paramaters at.
            min_norm_constraint: Minimum value for the norm constraint.
            max_norm_constraint: Maximum value for the norm constraint.
            sort_params: Whether to sort the parameters of the model or not.
            eps: Epsilon value for the numerical stability.
        """
        assert params is not None, "params parameter input to constrain_params function should not be empty."
        assert updates is not None, "updates parameter input to constrain_params function should not be empty."

        if sort_params:
            sorted_params = sorted(params.iteritems())
        else:
            if not isinstance(params, OrderedDict):
                warnings.warn("Parameters is not an ordered dictionary, this may cause inconsistent results.")

        for param in sorted_params:
            update_param = updates[param]
            clipped_param = self._clip_norms(update_param,
                                             max_norm_constraint,
                                             min_norm_constraint,
                                             eps)
            updates[param] = clipped_param
        return updates

    def constrain_param(self,
                        param=None,
                        max_norm_constraint=None,
                        min_norm_constraint=None,
                        eps=1e-7):
        """
            Apply the constraint directly on a specific parameter of the model.
            params: A dictionary of the name of paramaters that the constraint is going to be applied.
            min_norm_constraint: Minimum value for the norm constraint.
            max_norm_constraint: Maximum value for the norm constraint.
            eps: Epsilon value for the numerical stability.
        """

        assert param is not None, "params parameter input to constrain_params function should not be empty."
        clipped_param = self._clip_norms(param,
                                         max_norm_constraint,
                                         min_norm_constraint,
                                         eps)
        return clipped_param
