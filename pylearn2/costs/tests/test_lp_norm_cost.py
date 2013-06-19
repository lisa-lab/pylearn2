"""
Test LpNorm cost
"""
import os
from nose.tools import raises
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import MLP
from pylearn2.costs.cost import LpNorm
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train


def test_shared_variables():
    '''
    LpNorm should handle shared variables.
    '''
    model = MLP(
        layers=[Linear(dim=100, layer_name='linear', irange=1.0),
                Softmax(n_classes=10, layer_name='softmax', irange=1.0)],
        batch_size=100,
        nvis=3072
    )

    dataset = CIFAR10(which_set='train')

    cost = LpNorm(variables=model.get_params(), p=2)

    algorithm = SGD(
        learning_rate=0.01,
        cost=cost, batch_size=100,
        monitoring_dataset=dataset,
        termination_criterion=EpochCounter(1)
    )

    trainer = Train(
        dataset=dataset,
        model=model,
        algorithm=algorithm
    )

    trainer.main_loop()

    assert False


def test_symbolic_expressions_of_shared_variables():
    '''
    LpNorm should handle symbolic expressions of shared variables.
    '''
    assert False


@raises(Exception)
def test_symbolic_variables():
    '''
    LpNorm should not handle symbolic variables
    '''
    assert True


if __name__ == '__main__':
    test_shared_variables()
    test_symbolic_expressions_of_shared_variables()
    test_symbolic_variables()
