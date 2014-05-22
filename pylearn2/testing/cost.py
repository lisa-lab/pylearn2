""" Simple costs to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.costs.cost import Cost
from pylearn2.space import NullSpace
from pylearn2.utils import CallbackOp
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping


class CallbackCost(Cost):
    """
    A Cost that runs callbacks on the data.  Returns the sum of the data
    multiplied by the sum of all model parameters as the cost.  The callback is
    run via the CallbackOp so the cost must be used to compute one of the
    outputs of your theano graph if you want the callback to get called.  The
    is cost is designed so that the SGD algorithm will result in in the
    CallbackOp getting evaluated.

    Parameters
    ----------
    data_callback : optional, callbacks to run on data.
        It is either a Python callable, or a tuple (possibly nested),
        in the same format as data_specs.
    data_specs : (space, source) pair specifying the format
        and label associated to the data.
    """
    def __init__(self, data_callbacks, data_specs):
        self.data_callbacks = data_callbacks
        self.data_specs = data_specs
        self._mapping = DataSpecsMapping(data_specs)

    def get_data_specs(self, model):
        return self.data_specs

    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        callbacks = self.data_callbacks

        cb_tuple = self._mapping.flatten(callbacks, return_tuple=True)
        data_tuple = self._mapping.flatten(data, return_tuple=True)

        costs = []
        for (callback, data_var) in safe_zip(cb_tuple, data_tuple):
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
        self.get_data_specs(model)[0].validate(data)
        return sum(param.sum() for param in model.get_params())

    def get_data_specs(self, model):
        # This cost does not need any data
        return (NullSpace(), '')


class SumOfOneHalfParamsSquared(Cost):
    """
    A cost that is just 0.5 * the sum of all parameters squared, so the gradient
    on every parameter is the parameter itself.
    """

    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        return 0.5 * sum((param**2).sum() for param in model.get_params())

    def get_data_specs(self, model):
        # This cost does not need any data
        return (NullSpace(), '')
