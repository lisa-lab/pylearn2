"""
Test term_1_0_l1_penalty
"""
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.mlp import MLP, Sigmoid
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay
from pylearn2.termination_criteria import And, EpochCounter, MonitorBased
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import Default, L1WeightDecay


def create_dataset():
    """
    Create a fake dataset to initiate the training
    """
    x = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20],
                  [0.3, 0.1, 0.8, 0.1, 0.2, 0.6, 0.83, 0.45, 0.0, 0.67, 0.3,
                   0.74, 0.8, 0.1, 0.2, 0.46, 0.83, 0.45, 0.0, 0.67]])

    y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
                  1, 0, 1, 0, 0, 0, 0, 0]).reshape(20, 1)

    x_train = x[:7, :]
    x_valid = x[7:, :]
    y_train = y[:7]
    y_valid = y[7:]

    return x_train, y_train, x_valid, y_valid


def test_correctness():
    """
    Test that the cost function works with float64
    """
    x_train, y_train, x_valid, y_valid = create_dataset()

    trainset = DenseDesignMatrix(X=np.array(x_train), y=y_train)
    validset = DenseDesignMatrix(X=np.array(x_valid), y=y_valid)

    n_inputs = trainset.X.shape[1]
    n_outputs = 1
    n_hidden = 10

    hidden_istdev = 4 * (6 / float(n_inputs + n_hidden)) ** 0.5
    output_istdev = 4 * (6 / float(n_hidden + n_outputs)) ** 0.5

    model = MLP(layers=[Sigmoid(dim=n_hidden, layer_name='hidden',
                                istdev=hidden_istdev),
                        Sigmoid(dim=n_outputs, layer_name='output',
                                istdev=output_istdev)],
                nvis=n_inputs, seed=[2013, 9, 16])

    termination_criterion = And([EpochCounter(max_epochs=1),
                                 MonitorBased(prop_decrease=1e-7,
                                 N=2)])

    cost = SumOfCosts([(0.99, Default()),
                       (0.01, L1WeightDecay({}))])

    algo = SGD(1e-1,
               update_callbacks=[ExponentialDecay(decay_factor=1.00001,
                                 min_lr=1e-10)],
               cost=cost,
               monitoring_dataset=validset,
               termination_criterion=termination_criterion,
               monitor_iteration_mode='even_shuffled_sequential',
               batch_size=2)

    train = Train(model=model, dataset=trainset, algorithm=algo)
    train.main_loop()


if __name__ == '__main__':
    test_correctness()
