__authors__ = 'Vincent Archambault-Bouffard, Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.utils import safe_izip

class Default(Cost):
    """
    The default Cost to use with an MLP.
    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def expr(self, model, data, **kwargs):
        assert type(data) in (list, tuple)
        assert len(data) == 2
        (X, Y) = data
        return model.cost_from_X(X, Y)

    def get_data_specs(self, model):
        data = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return [data, sources]


class WeightDecay(Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    """

    def __init__(self, coeffs):
        """
        coeffs: a list, one element per layer, specifying the coefficient
                to multiply with the cost defined by the squared L2 norm of the weights
                for each layer.

                Each element may in turn be a list, ie, for CompositeLayers.
        """
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):

        def wrapped_layer_cost(layer, coef):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coef==0.:
                    return 0.
                else:
                    raise NotImplementedError(str(type(layer))+" does not implement get_weight_decay.")


        layer_costs = [ wrapped_layer_cost(layer, coeff)
            for layer, coeff in safe_izip(model.layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

    def get_data_specs(self, model):
        return [None, None]

class L1WeightDecay(Cost):
    """
    coeff * sum(abs(weights))

    for each set of weights.

    """

    def __init__(self, coeffs):
        """
        coeffs: a list, one element per layer, specifying the coefficient
                to multiply with the cost defined by the L1 norm of the
                weights(lasso) for each layer.

                Each element may in turn be a list, ie, for CompositeLayers.
        """
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):

        layer_costs = [ layer.get_l1_weight_decay(coeff)
            for layer, coeff in safe_izip(model.layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_l1_penalty'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MLP_L1Penalty'

        assert total_cost.ndim == 0

        total_cost.name = 'l1_penalty'

        return total_cost

    def get_data_specs(self, model):
        return [None, None]

