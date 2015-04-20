"""
Tests for pylearn2.termination_criteria.__init__ functions and classes.
"""

from pylearn2.termination_criteria import EpochCounter

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.monitor import push_monitor
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

import numpy as np


def test_epoch_counter():
    """
    Test epoch counter with max_epochs={True,False}
    """

    def produce_train_obj(new_epochs, max_epochs, model=None):
        if model is None:
            model = MLP(layers=[Softmax(layer_name='y',
                                        n_classes=2,
                                        irange=0.)],
                        nvis=3)
        else:
            model = push_monitor(model, 'old_monitor',
                                 transfer_experience=True)

        dataset = DenseDesignMatrix(X=np.random.normal(size=(6, 3)),
                                    y=np.random.normal(size=(6, 2)))

        epoch_counter = EpochCounter(max_epochs=max_epochs,
                                     new_epochs=new_epochs)

        algorithm = SGD(batch_size=2,
                        learning_rate=0.1,
                        termination_criterion=epoch_counter)

        return Train(dataset=dataset, model=model, algorithm=algorithm)

    def test_epochs(epochs_seen, n):
        assert epochs_seen == n, \
            "%d epochs seen and should be %d" % (epochs_seen, n)

    # Tests for 5 new epochs
    train_obj = produce_train_obj(new_epochs=True, max_epochs=5)
    train_obj.main_loop()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 5)
    train_obj = produce_train_obj(new_epochs=True,
                                  max_epochs=5,
                                  model=train_obj.model)
    train_obj.main_loop()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 2 * 5)

    # Tests for 5 max epochs
    train_obj = produce_train_obj(new_epochs=False, max_epochs=5)
    train_obj.main_loop()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 5)
    train_obj = produce_train_obj(new_epochs=False,
                                  max_epochs=5,
                                  model=train_obj.model)
    train_obj.main_loop()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 5)

    # Tests for 0 new epochs
    train_obj = produce_train_obj(new_epochs=True, max_epochs=0)
    before_train = train_obj.model.get_weights()
    train_obj.main_loop()
    after_train = train_obj.model.get_weights()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 0)
    assert np.all(before_train == after_train)

    train_obj = produce_train_obj(new_epochs=True,
                                  max_epochs=0,
                                  model=train_obj.model)
    before_train = train_obj.model.get_weights()
    train_obj.main_loop()
    after_train = train_obj.model.get_weights()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 0)
    assert np.all(before_train == after_train)

    # Tests for 0 max epochs
    train_obj = produce_train_obj(new_epochs=False,
                                  max_epochs=0,
                                  model=train_obj.model)
    before_train = train_obj.model.get_weights()
    train_obj.main_loop()
    after_train = train_obj.model.get_weights()
    test_epochs(train_obj.model.monitor.get_epochs_seen(), 0)
    assert np.all(before_train == after_train)
