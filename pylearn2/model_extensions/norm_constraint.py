"""
Classes for constraining the norms of weight matrices.
"""
import numpy as np
import warnings

from theano import tensor as T

from pylearn2.model_extensions.model_extension import ModelExtension
from pylearn2.utils import wraps


class ConstrainFilterL2Norm(ModelExtension):

    """
    Constrains the maximum L2 norm (not squared L2) of a
    weight matrix.

    Expects the weight matrix to either be the
    sole parameter of the model's `transformer` field or to
    be the model's `W` field.

    Parameters
    ----------
    limit : float or symbolic float
        The maximum norm of the weight matrix is constrained
        to be <= limit along the axes
    min_limit : float or symbolic float
        The minimum norm of the weight matrix is constrained
        to be => limit along the axes
    axis : int or tuple of int
        The axis or axes over which the norm is computed. Default is 0.
    """

    def __init__(self, limit, min_limit=0., axis=0):
        self.max_limit = limit
        self.min_limit = min_limit
        if (limit is not None) and (min_limit is not None):
            if limit < min_limit:
                raise ValueError('The maximum limit must be higher than '
                                 'the minimum limit.')
        self.axis = axis

    @wraps(ModelExtension.post_modify_updates)
    def post_modify_updates(self, updates, model):
        if hasattr(model, 'W'):
            W = model.W
        else:
            if not hasattr(model, 'transformer'):
                raise TypeError("model has neither 'W' nor 'transformer'.")
            transformer = model.transformer
            params = transformer.get_params()
            if len(params) != 1:
                raise TypeError("self.transformer does not have exactly one "
                                "parameter tensor.")
            W, = params

        if W in updates:
            updated_W = updates[W]
            l2_norms = T.sqrt(
                T.square(updated_W).sum(
                    axis=self.axis, keepdims=True
                )
            )
            if self.min_limit is None:
                min_limit = 0.
            else:
                min_limit = self.min_limit

            if self.max_limit is None:
                max_limit = l2_norms.max()
            else:
                max_limit = self.max_limit

            desired_norms = T.clip(l2_norms, min_limit, max_limit)
            scale = desired_norms / T.maximum(1e-7, l2_norms)
            updates[W] = updated_W * scale


class MaxL2FilterNorm(ConstrainFilterL2Norm):

    """
    A copy of ConstrainFilterL2Norm, made to preserve the old class name.

    This name is deprecated.

    Parameters
    ----------
    args : list
        Passed on to the superclass.
    kwargs : dict
        Passed on to the superclass.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("MaxL2FilterNorm is deprecated and may be removed on or"
                      " after 2016-01-31. Use ConstrainFilterL2Norm.")
        super(MaxL2FilterNorm, self).__init__(*args, **kwargs)


class ConstrainFilterMaxNorm(ModelExtension):

    """
    Constrains the maximum max norm of a weight matrix.

    Expects the weight matrix to either be the
    sole parameter of the model's `transformer` field or to
    be the model's `W` field.

    Parameters
    ----------
    limit : float or symbolic float
        The maximum norm of the weight matrix is constrained
        to be <= limit
    min_limit : float or symbolic float
        The minimum norm of the weight matrix is constrained
        to be => limit
    """

    def __init__(self, limit, min_limit=None):
        self.max_limit = limit
        self.min_limit = min_limit
        if (limit is not None) and (min_limit is not None):
            if limit < min_limit:
                raise ValueError('The maximum limit must be higher than '
                                 'the minimum limit.')

    @wraps(ModelExtension.post_modify_updates)
    def post_modify_updates(self, updates, model):
        if hasattr(model, 'W'):
            W = model.W
        else:
            if not hasattr(model, 'transformer'):
                raise TypeError("model has neither 'W' nor 'transformer'.")
            transformer = model.transformer
            params = transformer.get_params()
            if len(params) != 1:
                raise TypeError("self.transformer does not have exactly one "
                                "parameter tensor.")
            W, = params

        if W in updates:
            updated_W = updates[W]
            if self.min_limit is None:
                min_limit = 0.
            else:
                min_limit = self.min_limit

            if self.max_limit is None:
                max_limit = np.inf
            else:
                max_limit = self.max_limit

            if self.min_limit is not None:
                # This would be a pretty weird feature to want but I put
                # the interface here for compatibility with the L2 norm
                # constraint class.
                raise NotImplementedError()

            if self.max_limit is not None:
                updates[W] = T.clip(updated_W, -self.max_limit, self.max_limit)
