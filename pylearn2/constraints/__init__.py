import warnings

import theano.tensor as T
from theano.compat.python2x import OrderedDict

from pylearn2.utils import wraps

class Constraints(object):
    """
    A plug and play style class to contain different types of constraints.
    !!!IMPORTANT!!!
    Order of the constraints added to that class is important.
    Because the constrain_params are passed into the function in a specific order.
    """
    def __init__(self, constraints=None):
        """
        .. todo::
            WRITEME
        """
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

    def add_constraint(self, constraint):
        """
        .. todo::
        """
        self.constraints.append(constraint)

    def get_constraints(self):
        """
        .. todo::
            WRITEME
        """
        return constraints

    def apply_constraints(self, constraint_params):
        """
        Function that applies the constraints with the specified parameters for each constraint.
        """
        assert constraint_params is not None, "constraint parameters list should not be empty."
        for constrain_param, constraint in zip(constraint_params, self.constraints):
            constraint.apply_constraint(**constrain_param)

class Constraint(object):
    """
    Base class for implementing different types of constraints.
    """
    def apply_constraint(self,
                         constrain_on=None,
                         updates=None,
                         min_constraint=None,
                         max_constraint=None):
        """
        A function that applies the constraint. This function is implemented with Weight Norm
        constraint in mind. We should make the interface more generic for other types of constraints.
        May be replace the current parameters with *args.

        Parameters
        ----------
        WRITEME

        Returns
        -------
        WRITEME
        """
        raise NotImplementedError()

class NormConstraint(Constraint):
    """
    A class implementing norm constraints. This class can be used to implement max/min norm
    constraints on a matrix, vector or a tensor such that if the norm constraint is not
    satisfied rescale the values along the given axes.

    Applying norm constraint on the parameters was first proposed in the following paper:
        Srebro, Nathan, and Adi Shraibman. "Rank, trace-norm and max-norm." Learning Theory.
        Springer Berlin Heidelberg, 2005. 545-560.
    But it is further popularized in neural networks with drop-out in the following publication:
        Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature
        detectors." arXiv preprint arXiv:1207.0580 (2012).
    """
    def __init__(self, axes=(0,), dimshuffle_pattern=None):
        """
        axes: determines which axes to apply the constraints on.
        dimshuffle_pattern: axes that we dimshuffle the tensor/matrix/vector along.
        """
        self.dimshuffle_pattern = dimshuffle_pattern
        self.axes = axes

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

        axes = self.axes if self.axes is not None else (0,)
        assert type(axes) is tuple, "axes you have provided to the norm constraint class, should be a tuple."
        assert max_norm is not None
        min_norm_constr = min_norm if min_norm is not None else 0
        dimshuffle_pattern = self.dimshuffle_pattern

        sqr_param = T.sqr(init_param)
        norm = T.sqrt(T.sum(sqr_param, axis=axes))
        desired_norm = T.clip(norm, min_norm_constr, max_norm)
        desired_norm_ratio = desired_norm / (eps + norm)

        if dimshuffle_pattern is not None:
            desired_norm_ratio = desired_norm_ratio.dimshuffle(dimshuffle_pattern)
        clipped_param = init_param * desired_norm_ratio

        return clipped_param

    def constrain_update(self,
                        param=None,
                        updates=None,
                        min_norm_constraint=None,
                        max_norm_constraint=None,
                        sort_params=True,
                        eps=1e-7):
        """
        Apply the constraint on the updates of the model.
            param: Weight Parameter that the constraint is going to be applied.
            updates: Updates that we are going to update our paramater at.
            min_norm_constraint: Minimum value for the norm constraint.
            max_norm_constraint: Maximum value for the norm constraint.
            sort_params: Whether to sort the parameters of the model or not.
            eps: Epsilon value for the numerical stability.
        """
        assert param is not None, "param parameter input to constrain_update function should not be empty."
        assert updates is not None, "updates parameter input to constrain_update function should not be empty."

        update_param = updates[param]
        clipped_param = self._clip_norms(update_param,
                                       max_norm_constraint,
                                       min_norm_constraint,
                                       eps)

        updates[param] = clipped_param
        return updates

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

    @wraps(Constraint.apply_constraint)
    def apply_constraint(self,
                         constrain_on=None,
                         updates=None,
                         min_constraint=None,
                         max_constraint=None):
        """
        The function that applies the constraints using the functions by using the constrain_param
        and the constrain_updates functions.
        """
        if updates is None:
            return self.constrain_param(constrain_on, max_constraint, min_norm_constraint)
        else:
            return self.constrain_update(constrain_on, updates, min_constraint, max_constraint)

