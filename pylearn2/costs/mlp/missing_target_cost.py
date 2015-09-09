"""
The MissingTargetCost class.
"""
__author__ = 'Vincent Archambault-Bouffard'

from functools import wraps

import theano.tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace


class MissingTargetCost(Cost):
    """
    Dropout but with some targets optionally missing. The missing target is
    indicated by a value of -1.

    Parameters
    ----------
    dropout_args : WRITEME
    """

    supervised = True

    def __init__(self, dropout_args=None):
        self.__dict__.update(locals())
        del self.self

    @wraps(Cost.expr)
    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        if self.dropout_args:
            Y_hat = model.dropout_fprop(X, **self.dropout_args)
        else:
            Y_hat = model.fprop(X)
        costMatrix = model.layers[-1].cost_matrix(Y, Y_hat)
        # This sets to zero all elements where Y == -1
        costMatrix *= T.neq(Y, -1)
        return model.cost_from_cost_matrix(costMatrix)

    @wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(),
                                model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
