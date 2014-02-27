import pylearn2
import pylearn2.models.mlp
from pylearn2.space import VectorSpace
import theano
import theano.tensor as T
import numpy as np

# use random data
dim = 1000
n = 1000
x = np.random.uniform(size=(n, dim))

# create MLP
i0 = VectorSpace(dim)
s0 = pylearn2.models.mlp.Tanh(layer_name='h0', dim=1000, sparse_init=25)
s1 = pylearn2.models.mlp.Sigmoid(layer_name='h1', dim=1000, sparse_init=25)
s2 = pylearn2.models.mlp.RectifiedLinear(layer_name='h2', dim=1000, \
                                         sparse_init=25)
mlp_inst = pylearn2.models.mlp.MLP(layers=[s0,s1], nvis=dim, \
                                   input_space=i0)

# define tensor variable
X = T.matrix("X")

# compute the jacobian with the new method
f = theano.function([X], [(mlp_inst.jacobian(X))], allow_input_downcast = True)
res = f(x)[0]
print res
print res.shape # (batch_size, output, input)

# comparison with scan
results, updates = theano.scan(lambda i:T.grad(mlp_inst.fprop(X).flatten()[i], \
                X), sequences = [T.arange(mlp_inst.fprop(X).flatten().shape[0])])
compute_jac = theano.function([X], [results], allow_input_downcast = True)

k = np.random.randint(x.shape[0])
one_example = x[k].reshape((1, x.shape[1]))

print "Quadratic error is:", ((f(one_example)[0] \
        - compute_jac(one_example)[0].transpose((1,0,2)))**2).sum()

print "Close enough."
