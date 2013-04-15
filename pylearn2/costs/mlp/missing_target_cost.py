__author__ = 'Vincent Archambault-Bouffard'

import theano.tensor as T
from pylearn2.costs.cost import Cost


class MissingTargetCost(Cost):
    """
    A cost when some targets are missing
    The missing target is indicated by a value of -1
    """
    supervised = True

    def __init__(self, dropout_args=None):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data):
        assert type(data) in (list, tuple)
        assert len(data) == 2
        (X, Y) = data
        if self.dropout_args:
            Y_hat = model.dropout_fprop(X, **self.dropout_args)
        else:
            Y_hat = model.fprop(X)
        costMatrix = model.layers[-1].cost_matrix(Y, Y_hat)
        costMatrix *= T.neq(Y, -1)  # This sets to zero all elements where Y == -1
        return model.cost_from_cost_matrix(costMatrix)
