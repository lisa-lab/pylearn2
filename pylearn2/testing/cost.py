""" Simple costs to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.costs.cost import Cost
from pylearn2.utils import CallbackOp
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import is_flat_specs


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

    def __init__(self, data_callbacks, data_specs):
        """
        data_callback: optional, callbacks to run on data.
            It is either a Python callable, or a flat tuple,
            in the same format as data_specs.
        data_specs: (space, source) pair specifying the format
            and label associated to the data.
        """
        self.data_callbacks = data_callbacks
        self.data_specs = data_specs
        assert is_flat_specs(data_specs)

    def get_data_specs(self, model):
        return self.data_specs

    def expr(self, model, data):
        if not isinstance(self.data_callbacks, tuple):
            callbacks = (self.data_callbacks,)
        else:
            callbacks = self.data_callbacks

        if not isinstance(data, tuple):
            assert len(callbacks) == 1
            data = (data,)

        costs = []
        for (callback, data_var) in safe_zip(callbacks, data):
            orig_var = data_var
            data_var = CallbackOp(callback)(data_var)
            assert len(data_var.owner.inputs) == 1
            assert orig_var is data_var.owner.inputs[0]

            costs.append(data_var.sum())

        # sum() will call theano.add on the symbolic variables
        cost = sum(costs)
        model_terms = sum([param.sum() for param in model.get_params()])
        cost = cost * model_terms
        return cost


class SumOfParams(Cost):
    """
    A cost that is just the sum of all parameters, so the gradient
    on every parameter is 1.
    """

    def expr(self, model, data):
        return sum(param.sum() for param in model.get_params())

    def get_data_specs(self, model):
        return (None, None)
