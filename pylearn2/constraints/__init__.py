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
        This is a function that applies the constraint. This function is implemented with weight norm
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
    This class is implementing norm constraints. It can be used to implement max/min norm
    constraints on a matrix, vector or on a tensor such that if the norm constraint computed
    across specific axis is not satisfied the values are rescaled along those axes.

    Applying norm constraint on the parameters was first proposed in the following paper:
        Srebro, Nathan, and Adi Shraibman. "Rank, trace-norm and max-norm." Learning Theory.
        Springer Berlin Heidelberg, 2005. 545-560.

    But its use is further popularized in neural networks literature with drop-out in the following publication:
        Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature
        detectors." arXiv preprint arXiv:1207.0580 (2012).
    """
    def __init__(self,
                 max_norm=None,
                 min_norm=None,
                 is_input_axis=True,
                 eps=1e-7):
        """
        Apply the norm constraint on the parameters. For feedforward
        layers, norm constraint are usually applied on weights,
        but for convolutional neural networks constraint is
        applied on filters(usually a tensor) along specific
        axes(usually input). For scaling the parameters
        to satisfy the specific constraint we have to multiply
        the parameters $\theta$ with a scale $\alpha$.
        $\alpha$ is collapsed along the axes of '''axes''' argument.

        Parameters
        ----------
        norm : float, optional
            The maximum norm of the parameters.
        is_input_axis : bool, optional
            This determines whether to perform the dimshuffle along is
            input axes or output axes. By default this has been set to True.
        max_norm : float, optional
            maximum constraint on the norm of the matrix/tensor.
        min_norm : float, optional
            minimum constraint on the norm of the matrix/tensor.
        eps : float
            Epsilon, a small value to be added to norm for numerical stability
            to ensure that denominator never becomes 0 (default = 1e-7).
        """
        self.is_input_axis = is_input_axis
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.eps = eps
        assert min_norm is None and max_norm is None, "%s's constructor expects " % (self.__class__.__name__) + \
                " either min_norm or max_norm."

    def _clip_norms(self,
                    constrain_on, axes,
                    eps=1e-7):
        """
        Parameters
        ----------
        constrain_on : Theano shared variable.
            Matrix/tensor that we are going to apply the constraint on.
        """
        assert axes is not None, "%s._clip_norms function expects" % (self.__class__.__name__) + \
            "axes argument to be provided."
        min_constraint = 0.0
        max_constraint = np.inf
        if self.max_norm is not None:
            max_constraint = self.max_norm
        if self.min_norm is not None:
            min_constraint = self.min_norm

        sqr_param = T.sqr(constrain_on)
        norm = T.sqrt(T.sum(sqr_param, axis=axes, keepdims=True))
        desired_norm = T.clip(norm, min_constraint, max_constraint)
        desired_norm_ratio = desired_norm / (self.eps + norm)
        clipped_param = constrain_on * desired_norm_ratio
        return clipped_param

    @wraps(Constraint.apply_constraint)
    def apply_constraint(self,
                         constrain_on,
                         axes=(0,), updates=None):
        """
        This function applies the constraints on constrain_on argument.
        If updates dictionary is specified, it will be updated with
        the new constrained value.

        Parameters
        ----------
        constrain_on : theano shared variable.
            Theano shared variable that the constraint is going to be
            applied on.
        axes : tuple, optional
            Axes to apply the norm constraint over. axes are determined
            by the layer. Default value of this argument is (0,).
        updates : dictionary, optional
            This argument is a dictionary of theano shared variables as
            keys and their new (updated) values-usually as theano
            symbolic expressions as its elements. This dictionary
            is being passed to the train function as its given argument.
        """

        if updates is None:
            updates = OrderedDict({})
            clipped_param = self._clip_norms(constrain_on, axes)
            updates[constrain_on] = clipped_param
            return updates
        else:
            assert constrain_on in updates, ("%s.apply_constraint function expects" %
            self.__class__.__name__) + "constrain_on argument to be in provided updates dictionary."

            update_param = updates[constrain_on]
            clipped_param = self._clip_norms(update_param, axes, updates)
            updates[constrain_on] = clipped_param
            return updates
