"""
Unit tests for ../sparse_dataset.py
"""

import numpy as np
from pylearn2.datasets.sparse_dataset import SparseDataset
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.train import Train
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.termination_criteria import EpochCounter
from scipy.sparse import csr_matrix
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils import sharedX
import theano.tensor as T


class SoftmaxModel(Model):

    def __init__(self, dim):
        self.dim = dim
        rng = np.random.RandomState([2014, 4, 22])
        self.P = sharedX(rng.uniform(-1., 1., (dim,)))
        self.force_batch_size = None

    def get_params(self):
        return [self.P]

    def get_input_space(self):
        return VectorSpace(self.dim)

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        assert X.ndim == 2
        return T.nnet.softmax(X*self.P)


class DummyCost(DefaultDataSpecsMixin, Cost):
    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        return T.square(model(X) - X).mean()


def test_iterator():
    """
    tests wether SparseDataset can be loaded and
    initializes iterator
    """

    x = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    ds = SparseDataset(from_scipy_sparse_dataset=x)
    it = ds.iterator(mode='sequential', batch_size=1)
    it.next()


def test_training_a_model():
    """
    tests wether SparseDataset can be trained
    with a dummy model.
    """

    dim = 3
    m = 10
    rng = np.random.RandomState([22, 4, 2014])

    X = rng.randn(m, dim)
    ds = csr_matrix(X)
    dataset = SparseDataset(from_scipy_sparse_dataset=ds)

    model = SoftmaxModel(dim)
    learning_rate = 1e-1
    batch_size = 5

    epoch_num = 2
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    init_momentum=None,
                    set_batch_size=False)

    train = Train(dataset, model, algorithm, save_path=None,
                  save_freq=0, extensions=None)

    try:
        train.main_loop()
    except:
        message = "Could not train a dummy SoftMax model with a sparce dataset"
        raise AssertionError(message)

if __name__ == '__main__':
    test_iterator()
    test_training_a_model()
