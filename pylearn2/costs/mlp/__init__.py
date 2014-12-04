"""
Costs for use with the MLP model class.
"""
__authors__ = 'Vincent Archambault-Bouffard, Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from functools import wraps
import operator
import warnings

from theano import tensor as T
from theano.compat.six.moves import reduce

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as


class Default(DefaultDataSpecsMixin, Cost):
    """The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return model.cost_from_X(data)

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class WeightDecay(NullDataSpecsMixin, Cost):
    """L2 regularization cost for MLP.

    coeff * sum(sqr(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(sqr(weights))
            added up for each set of weights.
        """
        self.get_data_specs(model)[0].validate(data)
        assert T.scalar() != 0.  # make sure theano semantics do what I want

        def wrapped_layer_cost(layer, coeff):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coeff == 0.:
                    return 0.
                else:
                    reraise_as(NotImplementedError(str(type(layer)) +
                                                   " does not implement "
                                                   "get_weight_decay."))

        if isinstance(self.coeffs, list):
            warnings.warn("Coefficients should be given as a dictionary "
                          "with layer names as key. The support of "
                          "coefficients as list would be deprecated from "
                          "03/06/2015")
            layer_costs = [wrapped_layer_cost(layer, coeff)
                           for layer, coeff in safe_izip(model.layers,
                                                         self.coeffs)]
            layer_costs = [cost for cost in layer_costs if cost != 0.]
        else:
            layer_costs = []
            for layer in model.layers:
                layer_name = layer.layer_name
                if layer_name in self.coeffs:
                    cost = wrapped_layer_cost(layer, self.coeffs[layer_name])
                    if cost != 0.:
                        layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class L1WeightDecay(NullDataSpecsMixin, Cost):
    """L1 regularization cost for MLP.

    coeff * sum(abs(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(abs(weights))
            added up for each set of weights.
        """

        assert T.scalar() != 0.  # make sure theano semantics do what I want
        self.get_data_specs(model)[0].validate(data)
        if isinstance(self.coeffs, list):
            warnings.warn("Coefficients should be given as a dictionary "
                          "with layer names as key. The support of "
                          "coefficients as list would be deprecated "
                          "from 03/06/2015")
            layer_costs = [layer.get_l1_weight_decay(coeff)
                           for layer, coeff in safe_izip(model.layers,
                                                         self.coeffs)]
            layer_costs = [cost for cost in layer_costs if cost != 0.]

        else:
            layer_costs = []
            for layer in model.layers:
                layer_name = layer.layer_name
                if layer_name in self.coeffs:
                    cost = layer.get_l1_weight_decay(self.coeffs[layer_name])
                    if cost != 0.:
                        layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.as_tensor_variable(0.)
            rval.name = '0_l1_penalty'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_L1Penalty'

        assert total_cost.ndim == 0

        total_cost.name = 'l1_penalty'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False
