import numpy as np
from theano import config
from theano import function
from theano import tensor

from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.sandbox.rnn.models.mlp import Recurrent
from pylearn2.space import VectorSpace

mlp = MLP(layers=[Tanh(dim=25, layer_name='pre_rnn', irange=0.01),
                  Recurrent(dim=50, layer_name='recurrent', irange=0.01),
                  Tanh(dim=100, layer_name='h', irange=0.01)],
                  input_space=SequenceSpace(VectorSpace(dim=25)))

# Very simple test
input = tensor.tensor3()
output = mlp.fprop(input)
f = function([input], output)

print f(np.random.rand(10, 5, 25).astype(config.floatX))