import warnings

import theano.tensor as T
from theano.compat.python2x import OrderedDict

from pylearn2.utils import wraps, get_dimshuffle_pattern

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
        return self.constraints

    def apply_constraints(self, constraint_args, input_axes=None, output_axes=None):
        """
        Function that applies the constraints with the specified parameters for each constraint.

        Parameters
        ----------
        constraint_args: list of dictionaries.
            A list of function arguments(in a dictionary) to pass to apply_constraints function
            of each constraint.
        input_axes: WRITEME
        output_axes: WRITEME
        """
        assert constraint_args is not None, "constraint parameters list should not be empty."
        for constraint_arg, constraint in zip(constraint_args, self.constraints):
            if constraint.is_input_axes:
                constraint_arg["axes"] = input_axes
            else:
                constraint_arg["axes"] = output_axes
            constraint.apply_constraint(**constraint_arg)


    def is_empty(self):
        """
        .. todo::
            WRITEME
        """
        if self.constraints is None:
            return False
        else:
            return len(self.constraints) == 0

class Constraint(object):
    """
    Base class for implementing different types of constraints.
    """
    def apply_constraint(self,
                         constrain_on, axes, updates,
                         min_constraint=None, max_constraint=None):
        """
        A function that applies the constraint. This function is implemented with weight norm
        constraint in mind. We should make the interface more generic for other types of constraints.

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
    satisfied the values are rescaled along the given axes.

    Applying norm constraint on the parameters was first proposed in the following paper:
        Srebro, Nathan, and Adi Shraibman. "Rank, trace-norm and max-norm." Learning Theory.
        Springer Berlin Heidelberg, 2005. 545-560.

    But it is further popularized in neural networks with drop-out in the following publication:
        Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature
        detectors." arXiv preprint arXiv:1207.0580 (2012).
    """
    def __init__(self, min_norm=None, max_norm=None,
                 is_input_axes=True):
        """
        This class applies the norm constraint on the parameters. For feedforward layers, norm constraint are
        usually applied on weights, but for convolutional neural networks constraint is being
        applied on filters(usually a tensor) along specific axes. For scaling the parameters to satisfy the specific
        constraint we have to multiply the parameters $\theta$ with a scale $\alpha$. $\alpha$ is
        collapsed along the axes of '''axes''' argument. In order to make elementwise
        multiplication valid, $\alpha$ should be dimshuffled along the dimshuffle_pattern axes.
        TODO
        Automatically generate dimshuffle_pattern instead of relying on user's input.

        Parameters
        ----------
        min_norm : float, optional
            The minimum norm of the parameters.
        max_norm : float, optional
            The maximum norm of the parameters.
        is_input_axes: bool, optional
            This determines whether to perform the dimshuffle along is input axes or output
            axes. By default this has been set to True.
        """
        self.is_input_axes = is_input_axes
        self.min_norm = min_norm
        self.max_norm = max_norm

    def _clip_norms(self, init_param, axes,
                    max_norm, min_norm,
                    eps):
        """
        Parameters
        ----------
        init_param : Theano shared variable.
            The parameter that we are going to apply the constraint on.
        max_norm : float
            Maximum norm constraint.
        min_norm : float
            Minimum norm constraint.
        eps : float
            Epsilon, a small value to be added to norm for numerical stability to ensure that
        denominator never becomes 0.
        """
        assert max_norm is not None or min_norm  is not None, "%s._clip_norms function expects either min_norm or max_norm argument to be provided." % (self.__class__.__name__)
        assert axes is not None, "%s._clip_norms function expects axes argument to be provided." % (self.__class__.__name__)

        min_norm_constr = min_norm if min_norm is not None else 0

        param_ndim = init_param.ndim
        dimshuffle_pattern = get_dimshuffle_pattern(axes, param_ndim)

        sqr_param = T.sqr(init_param)
        norm = T.sqrt(T.sum(sqr_param, axis=axes))
        desired_norm = T.clip(norm, min_norm_constr, max_norm)
        desired_norm_ratio = desired_norm / (eps + norm)

        if dimshuffle_pattern is not None:
            desired_norm_ratio = desired_norm_ratio.dimshuffle(dimshuffle_pattern)

        clipped_param = init_param * desired_norm_ratio
        return clipped_param

    def constrain_update(self, param, axes, updates,
                         min_norm=None, max_norm=None,
                         eps=1e-7):
        """
        Apply the constraint on the updates of the model.

        Parameters
        ----------
        param : Theano shared variable
            Weight parameter that the constraint is going to be applied.
        axes : tuple
            The axes to apply the norm constraint over. axes are determined by the layer.
        updates : dictionary
            Updates that we are going to update our parameter at.
        min_norm : float, optional
            Minimum value for the norm constraint.
        max_norm : float, optional
            Maximum value for the norm constraint.
        eps : float, optional
            Epsilon value for the numerical stability.
        """
        assert param is not None, "param parameter input to constrain_update function should not be empty."
        assert updates is not None, "updates parameter input to constrain_update function should not be empty."

        if min_norm is None:
            min_norm = self.min_norm
        else:
            self.min_norm = self.min_norm

        if max_norm is None:
            max_norm = self.max_norm
        else:
            self.max_norm = max_norm

        update_param = updates[param]
        clipped_param = self._clip_norms(update_param, axes,
                                         max_norm, min_norm,
                                         eps)

        updates[param] = clipped_param
        return updates

    def constrain_updates(self, params, axes, updates,
                          min_norm=None, max_norm=None,
                          sort_params=True, eps=1e-7):
        """
        Apply the constraint on the updates dictionary of the model. This function applies the constraint
        on all the parameters in an ordered params dictionary.

        Parameters
        ----------
        params : dictionary
            A dictionary of the name of parameters that the constraint is going to be applied.
        axes : tuple
            The axes to apply the norm constraint over. axes are determined by the layer.
        updates : dictionary
            Updates that we are going to update our parameters at.
        min_norm : float
            Minimum value for the norm constraint.
        max_norm : float
            Maximum value for the norm constraint.
        sort_params : bool
            Whether to sort the parameters of the model or not.
        eps : float
            Epsilon value for the numerical stability.
        """
        assert params is not None, "params parameter input to constrain_params function should not be empty."
        assert updates is not None, "updates parameter input to constrain_params function should not be empty."
        if sort_params:
            sorted_params = sorted(params.iteritems())
        else:
            if not isinstance(params, OrderedDict):
                warnings.warn("Parameters to %s.constrain_updates function is not an ordered\
                        dictionary,this may cause inconsistent results." % (self.__class__.__name__))
            sorted_params = params.iteritems()

        for key, param in sorted_params:
            updates = self.constrain_update(param, updates, axes,
                                            min_norm, max_norm,
                                            eps)
        return updates

    def constrain_param(self, param, axes,
                        min_norm=None, max_norm=None,
                        eps=1e-7):
        """
        Apply the constraint directly on a specific parameter of the model.

        Parameters
        ----------
        params : dictionary
            A dictionary of the name of parameters that the constraint is going to be applied.
        axes : tuple
            The axes to apply the norm constraint over. axes are determined by the layer.
        min_norm : float, optional
            Minimum value for the norm constraint.
        max_norm : float, optional
            Maximum value for the norm constraint.
        eps : float, optional
            Epsilon value for the numerical stability.
        """

        assert param is not None, "params parameter input to constrain_params function should not be empty."

        clipped_param = self._clip_norms(param, axes,
                                         max_norm, min_norm,
                                         eps)
        return clipped_param

    @wraps(Constraint.apply_constraint)
    def apply_constraint(self, constrain_on, axes, updates=None,
                         min_constraint=None, max_constraint=None):
        """
        The function that applies the constraints using the functions by using the constrain_param
        and the constrain_updates functions.

        Parameters
        ----------
        constrain_on : theano shared variable.
            Theano shared variable that the constraint is going to be applied on.
        axes : tuple
            Axes to apply the norm constraint over. axes are determined by the layer.
        updates : dictionary, optional
            update dictionary that is being passed to the train function.
        min_constraint : float, optional
            minimum value that the constraint should satisfy
        max_constraint : float, optional
            maximum value that the constraint should satisfy
        """
        if updates is None:
            return self.constrain_param(constrain_on, axes, min_constraint, max_constraint)
        else:
            return self.constrain_update(constrain_on, axes, updates, min_constraint, max_constraint)
