""" Simple costs to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.costs.cost import Cost
from pylearn2.utils import CallbackOp

class CallbackCost(Cost):
    """
    A Cost that runs callbacks on the data.
    Returns the sum of the data multiplied by the
    sum of all model parameters as the cost.
    The callback is run via the CallbackOp
    so the cost must be used to compute one
    of the outputs of your theano graph if you
    want the callback to get called.
    The is cost is designed so that the SGD algorithm
    will result in in the CallbackOp getting
    evaluated.
    """

    def __init__(self, X_callback = None, y_callback = None,
            supervised = False):
        """
            X_callback: optional, callback to run on X
            y_callback: optional, callback to run on y
            supervised: whether this is a supervised cost or not

            (It is possible to be a supervised cost and not
            run callbacks on y, but it is not possible to be
            an unsupervised cost and run callbacks on y)
        """
        self.__dict__.update(locals())
        del self.self

        if not supervised:
            assert y_callback is None

    def __call__(self, model, X, Y = None):

        if self.X_callback is not None:
            orig_X = X
            X = CallbackOp(self.X_callback)(X)
            assert len(X.owner.inputs) == 1
            assert orig_X is X.owner.inputs[0]

        if self.y_callback is not None:
            Y = CallbackOp(self.y_callback)(Y)

        cost = X.sum()
        if self.supervised and Y is not None:
            cost = cost + Y.sum()

        model_terms = sum([param.sum() for param in model.get_params()])

        cost = cost * model_terms

        return cost

class SumOfParams(Cost):
    """
    A cost that is just the sum of all parameters, so the gradient
    on every parameter is 1.
    """

    def __call__(self, model, X, Y = None):
        assert Y is None

        return sum(param.sum() for param in model.get_params())
