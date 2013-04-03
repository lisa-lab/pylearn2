__author__ = 'Vincent Archambault-Bouffard'

import theano.tensor as T
from pylearn2.costs.cost import Cost


class MissingTargetCost(Cost):
    """
    A cost when some targets are missing
    The missing target is indicated by a value of -1
    """
    supervised = True

    def __call__(self, model, X, Y):
        Y_hat = model.fprop(X, apply_dropout=model.use_dropout)
        costMatrix = model.layers[-1].cost_matrix(Y, Y_hat)
        costMatrix *= T.neq(Y, -1)  # This sets to zero all elements where Y == -1
        return model.cost_from_cost_matrix(costMatrix)