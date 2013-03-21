__author__ = 'Vincent Archambault-Bouffard'

import theano.tensor as T
from pylearn2.costs.cost import Cost


class MeanSquareMissingValueCost(Cost):
    """
    Computes the mean square value error
    Skip the missing features (indicated a -1 in Y)
    """
    supervised = True

    def __call__(self, model, X, Y):
        T.pow() - Y
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()