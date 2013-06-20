import numpy as np
import theano
from pylearn2.linear.initializer import *

def test_initializer_subclasses():
    shape = (30, 31)
    mask_weights = np.random.randint(0,2,shape)
    biases = np.random.randn(31)
    initializers = [ \
        Uniform(init_range = 0.5),
        Uniform(init_range = (0.3,0.2)),
        Uniform(init_range = 0.05, biases = biases),
        Uniform(init_range = 1.0, biases = 1),
        Uniform(init_range = 0.1, mask_weights = mask_weights),
        Normal(stdev = 0.05),
        Normal(stdev = 0.05, mean = 1.0),
        Normal(stdev = 0.01, biases = biases),
        Normal(stdev = 0.05, biases = 1),
        Normal(stdev = 0.05, mask_weights = mask_weights),
        Sparse(),
        Sparse(sparse_init = 20, stdev = 1.0),
        Sparse(biases = biases),
        Sparse(biases = 1),
        Sparse(sparse_init = 29, mask_weights = mask_weights),
        Instance(np.random.randn(30,31)),
        Instance(np.random.randn(30,31), biases = biases),
        Instance(np.random.randn(30,31), biases = 1),
        Instance(np.random.randn(30,31), mask_weights = mask_weights) \
    ]
    rng = np.random.RandomState(9009)
    for initializer in initializers:
        assert initializer.get_weights(rng, shape).shape == shape
        assert initializer.get_biases(rng, shape).shape == (shape[-1],)
        mask = initializer.get_mask()
        if isinstance(mask, int):
            assert (mask == 1)
        else:
            assert (mask.shape == mask_weights.shape)

def test_bad_arguments():
    rng = np.random.RandomState(9009)
    raised = False
    try:
        s = Sparse(sparse_init=12).get_weights(rng, shape=(10, 12))
    except AssertionError:
        raised = True
    assert raised
    
    # tests mask_weights.shape == shape
    shape = (30, 31)
    invalid_shape = (30,30)
    mask_weights = np.random.randint(0,2,invalid_shape)
    initializers = [\
        Uniform(0.05, mask_weights = mask_weights),
        Normal(0.05, mask_weights = mask_weights),
        Sparse(mask_weights = mask_weights)\
    ]
    for initializer in initializers:
        raised = False
        try:
            initializer.get_weights(rng, shape)
        except ValueError:
            raised = True
        assert raised
    
    # instance can detect invalid mask_weights.shape in constructor:
    raised = False
    try:
        i = Instance(np.random.randn(30,31), mask_weights=mask_weights)
    except ValueError:
        raised = True
    assert raised
    
