import theano.tensor as T
import warnings
from error import SupervisedError

class CrossEntropy(SupervisedError):
    def __call__(self, model, X, Y):
        return (
            - Y * T.log(model(X)) -
            (1 - Y) * T.log(model(X))
            ).sum(axis=1).mean()

