import theano.tensor as T
import warnings
from pylearn2.costs.cost import SupervisedCost


class CrossEntropy(SupervisedCost):
    def __call__(self, model, X, Y):
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()
