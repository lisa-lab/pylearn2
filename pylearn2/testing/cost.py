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
    Returns the sum of the data as the cost.
    The callback is run via the CallbackOp
    so the cost must be used to compute one
    of the outputs of your theano graph if you
    want the callback to get called.
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

    def __call__(self, X, y = None):

        if self.X_callback is not None:
            X = CallbackOp(self.X_callback)(X)

        if self.y_callback is not None:
            y = CallbackOp(self.y_callback)(y)

        cost = X.sum()
        if y is not None:
            cost = cost + y.sum()

        return cost

