__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.train import Train
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm

class DummyModel(Model):
    def  __init__(self, num_features):
        self.input_space = VectorSpace(num_features)

class DummyAlgorithm(TrainingAlgorithm):
    pass

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
                 save_freq=1, callbacks=None)

    try:
        train.main_loop()
    except RuntimeError:
        return
    assert False # train did not complain, this is a bug
