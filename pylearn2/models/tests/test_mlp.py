import numpy as np
import theano
from pylearn2.models.mlp import MLP, Linear


def test_masked_fprop():
    # Construct a dirt-simple linear network with identity weights.
    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0), Linear(2, 'h1', irange=0)])
    mlp.layers[0].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[1].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[0].set_biases(np.arange(1, 3, dtype=mlp.get_weights().dtype))
    mlp.layers[1].set_biases(np.arange(3, 5, dtype=mlp.get_weights().dtype))

    # Verify that get_total_input_dimension works.
    np.testing.assert_equal(mlp.get_total_input_dimension(['h0', 'h1']), 4)
    inp = theano.tensor.matrix()

    # Accumulate the sum of output of all masked networks.
    l = []
    for mask in xrange(16):
        l.append(mlp.masked_fprop(inp, mask))
    outsum = reduce(lambda x, y: x + y, l)

    f = theano.function([inp], outsum)
    np.testing.assert_equal(f([[5, 3]]), [[144., 144.]])
    np.testing.assert_equal(f([[2, 7]]), [[96., 208.]])

    # Verify that using a too-wide mask fails.
    raised = False
    try:
        mlp.masked_fprop(inp, 22)
    except ValueError:
        raised = True
    np.testing.assert_(raised)


if __name__ == "__main__":
    test_masked_fprop()
