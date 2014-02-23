import pylearn2
import pylearn2.datasets.mnist
import pylearn2.models.mlp
from pylearn2.space import VectorSpace
import theano
import theano.tensor as T
import numpy as np

# use MNIST data
mnist = pylearn2.datasets.mnist.MNIST("train", False, False, \
                                      True, start=0, stop=1000)
mnist_input = mnist.get_data()[0]

# create MLP
i0 = VectorSpace(mnist_input.shape[1])
s0 = pylearn2.models.mlp.Tanh(layer_name='h0', dim=1000, sparse_init=25)
s1 = pylearn2.models.mlp.Sigmoid(layer_name='h1', dim=1000, sparse_init=25)
s2 = pylearn2.models.mlp.RectifiedLinear(layer_name='h2', dim=1000, \
                                         sparse_init=25)
mlp_inst = pylearn2.models.mlp.MLP(layers=[s0,s1], nvis=mnist_input.shape[1], \
                                   input_space=i0)

# define tensor variable
X = T.matrix("X")

# compute the jacobian with the new method
f = theano.function([X], [(mlp_inst.jacobian(X))])
mnist_minibatch = mnist_input[:100]
res = f(mnist_minibatch)[0]
print res
print res.shape # (batch_size, output, input)

# comparison with scan
results, updates = theano.scan(lambda i:T.grad(mlp_inst.fprop(X).flatten()[i], \
                X), sequences = [T.arange(mlp_inst.fprop(X).flatten().shape[0])])
compute_jac = theano.function([X], [results], allow_input_downcast = True)

k = np.random.randint(mnist_minibatch.shape[0])
one_digit = mnist_minibatch[k].reshape((1, mnist_minibatch.shape[1]))

print "Quadratic error is:", ((f(one_digit)[0] \
        - compute_jac(one_digit)[0].transpose((1,0,2)))**2).sum()

print "Close enough."
