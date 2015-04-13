"""
Test LpPenalty cost
"""
import numpy
import theano
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import MLP
from pylearn2.costs.cost import LpPenalty


def test_correctness():
    """
    Test that the Theano implementation matches a numpy implementation.
    """
    model = MLP(
        layers=[Linear(dim=10, layer_name='linear', irange=1.0),
                Softmax(n_classes=2, layer_name='softmax', irange=1.0)],
        batch_size=10,
        nvis=10
    )

    cost = LpPenalty(variables=model.get_params(), p=2)

    penalty = cost.expr(model, None)

    penalty_function = theano.function(inputs=[], outputs=penalty)

    p = penalty_function()

    actual_p = 0
    for param in model.get_params():
        actual_p += numpy.sum(param.get_value() ** 2)

    assert numpy.allclose(p, actual_p)


if __name__ == '__main__':
    test_correctness()
