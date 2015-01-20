"""
Classes for constraining the norms of weight matrices.
"""
from pylearn2.utils import wraps

from theano import tensor as T

from pylearn2.model_extensions.model_extension import ModelExtension


class MaxL2FilterNorm(ModelExtension):

    """
    Constrains the maximum L2 norm (not squared L2) norm of a
    weight matrix.

    Expects the weight matrix to either be the
    sole parameter of the model's `transformer` field or to
    be the model's `W` field.

    Parameters
    ----------
    limit : float or symbolic float
        The maximum column norm of the weight matrix is constrained
        to be <= limit
    """

    def __init__(self, limit):
        self.limit = limit

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
            col_norms = T.sqrt(T.square(W).sum(axis=0))
            desired_norms = T.minimum(col_norms, self.limit)
            scale = desired_norms / T.maximum(1e-7, col_norms)
            updates[W] = updated_W * scale
