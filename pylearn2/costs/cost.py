""" Classes representing loss functions.
Currently, these are primarily used to specify
the objective function for the SGD algorithm."""
import theano.tensor as T

class Cost(object):
    """Abstract class representing a loss function"""
    pass

class SupervisedCost(object):
    """Abstract class representing a cost of both features and labels"""

    def __call__(self, model, X, Y):
        """
            model: the model the cost is applied to
            X: a batch of features. First axis indexes examples
            Y: a batch of labels corresponding to X

            Returns a symbolic expression for the loss function.
        """
        raise NotImplementedError(str(self)+" does not implement __call__")


class UnsupervisedCost(object):
    """Abstract class representing a cost of features only"""

    def __call__(self, model, X):
        """
            model: the model the cost is applied to
            X: a batch of features. First axis indexes examples

            Returns a symbolic expression for the loss function.
        """
        raise NotImplementedError(str(self)+" does not implement __call__")

class CrossEntropy(SupervisedCost):
    """WRITEME"""
    def __call__(self, model, X, Y):
        """WRITEME"""
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()
