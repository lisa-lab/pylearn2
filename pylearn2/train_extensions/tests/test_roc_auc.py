from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_sklearn
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.mlp import MLP, Softmax, Sigmoid
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
import numpy as np
import unittest

def get_random_dataset(shape):
    """Generate random features and targets."""
    X = np.random.random(shape)
    y = np.random.randint(2, size=shape[0])
    y = np.vstack((y, 1-y)).T # one-hot encoding
    dataset = DenseDesignMatrix(X=X, y=y)
    return dataset

class TestROCAUCChannel(unittest.TestCase):
    """Train a simple model and calculate ROC AUC on the monitoring datasets."""
    def test_roc_auc(self):
        skip_if_no_sklearn()
        from pylearn2.train_extensions.roc_auc import ROCAUCChannel
        train_dataset = get_random_dataset((1000, 30))
        valid_dataset = get_random_dataset((500, 30))
        test_dataset = get_random_dataset((1500, 30))
        model = MLP(nvis=30,
                    layers=[Sigmoid(layer_name='h0', dim=30, sparse_init=15),
                            Softmax(layer_name='y', n_classes=2, irange=0.)])
        algorithm = SGD(learning_rate=1e-3, batch_size=100,
                        monitoring_dataset={'train': train_dataset,
                                            'valid': valid_dataset,
                                            'test': test_dataset},
                        termination_criterion=EpochCounter(10))
        train = Train(dataset=train_dataset, model=model, algorithm=algorithm,
                      extensions=[ROCAUCChannel()])
        train.main_loop()
