import warnings

from itertools import izip

import numpy as np

import theano.tensor as T
from theano.compat.python2x import OrderedDict

from pylearn2.utils import wraps


class Constraint(object):
    """
    Base class for implementing different types of constraints.
    """
    def apply_constraint(self,
                         constrain_on, axes,
                         updates=None):
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
    def __init__(self, norm=None,
                 is_input_axis=True,
                 is_max_constraint=True,
                 eps=1e-7):
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
        norm : float, optional
            The maximum norm of the parameters.
        is_input_axis : bool, optional
            This determines whether to perform the dimshuffle along is input axes or output
            axes. By default this has been set to True.
        is_max_constraint : bool, optional
            is_max_constraint is a flag that determines whether to apply constraint as a max norm
            constraint or min norm constraint. By default this is True.
        eps : float
            Epsilon, a small value to be added to norm for numerical stability to ensure that
            denominator never becomes 0 (default = 1e-7).
        """
        self.is_input_axis = is_input_axis
        self.is_max_constraint = is_max_constraint
        self.norm = norm
        self.eps = eps
        assert norm is not None, "%s's constructor expects " % (self.__class__.__name__) + \
                "norm argument to be provided."

    def _clip_norms(self, constrain_on, axes,
                    eps=1e-7):
        """
        Parameters
        ----------
        init_param : Theano shared variable.
            The parameter that we are going to apply the constraint on.
        """
        assert axes is not None, "%s._clip_norms function expects" % (self.__class__.__name__) + \
            "axes argument to be provided."
        min_constraint = 0.0
        max_constraint = np.inf
        if self.is_max_constraint:
            max_constraint = self.norm
        else:
            min_constraint = self.norm
        sqr_param = T.sqr(constrain_on)
        norm = T.sqrt(T.sum(sqr_param, axis=axes, keepdims=True))
        desired_norm = T.clip(norm, min_constraint, max_constraint)
        desired_norm_ratio = desired_norm / (self.eps + norm)
        clipped_param = init_param * desired_norm_ratio
        return clipped_param

    @wraps(Constraint.apply_constraint)
    def apply_constraint(self, constrain_on, axes,
                         updates=None):
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
        """
        if updates is None:
            clipped_param = self._clip_norms(constrain_on, axes)
            return self.constrain_param(constrain_on, axes)
        else:
            update_param = updates[constrain_on]
            clipped_param = self._clip_norms(update_param, axes, updates)
            updates[constrain_on] = clipped_param
            return updates

