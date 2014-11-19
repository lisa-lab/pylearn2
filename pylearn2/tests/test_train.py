__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from types import MethodType
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.train import Train
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.train_extensions import TrainExtension
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

class DummyModel(Model):

    _params = []

    def  __init__(self, num_features):
        super(DummyModel, self).__init__()
        self.input_space = VectorSpace(num_features)

class DummyAlgorithm(TrainingAlgorithm):
    pass

class ParamMonitor(TrainExtension):
    """
    Mock train extension for monitoring model params on monitor and on save
    """

    def on_save(self, model, dataset, algorithm):
        """
        Store state of parameters, ensure that parameters are the same
        as when the model was last monitored
        """
        self.params_on_save = np.asarray(model.get_param_values())
        param_pairs = zip(self.params_on_save, self.params_on_monitor)
        for save_params, monitor_params in param_pairs:
            assert np.array_equal(save_params, monitor_params)

    def on_monitor(self, model, dataset, algorithm):
        """
        Store state of parameters
        """
        self.params_on_monitor = np.asarray(model.get_param_values())

def only_run_extensions(self):
    for extension in self.extensions:
        extension.on_save(self.model, self.dataset, self.algorithm)

def test_execution_order():

    # ensure save is called directly after monitoring by checking 
    # parameter values in `on_monitor` and `on_save`.

    model = MLP(layers=[Softmax(layer_name='y',
                                n_classes=2,
                                irange=0.)],
                nvis=3)

    dataset = DenseDesignMatrix(X=np.random.normal(size=(6, 3)),
                                y=np.random.normal(size=(6, 2)))

    epoch_counter = EpochCounter(max_epochs=1)

    algorithm = SGD(batch_size=2, learning_rate=0.1,
                    termination_criterion=epoch_counter)

    extension = ParamMonitor()

    train = Train(dataset=dataset,
                  model=model,
                  algorithm=algorithm,
                  extensions=[extension],
                  save_freq=1,
                  save_path="save.pkl")

    # mock save
    train.save = MethodType(only_run_extensions, train)

    train.main_loop()

def test_serialization_guard():

    # tests that Train refuses to serialize the dataset

    dim = 2
    m = 11

    rng = np.random.RandomState([28,9,2012])
    X = rng.randn(m, dim)
    dataset = DenseDesignMatrix(X=X)

    model = DummyModel(dim)
    # make the dataset part of the model, so it will get
    # serialized
    model.dataset = dataset

    Monitor.get_monitor(model)

    algorithm = DummyAlgorithm()

    train = Train(dataset, model, algorithm, save_path='_tmp_unit_test.pkl',
                 save_freq=1, extensions=None)

    try:
        train.main_loop()
    except RuntimeError:
        return
    assert False # train did not complain, this is a bug
