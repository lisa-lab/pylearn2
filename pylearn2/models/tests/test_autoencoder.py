"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from pylearn2.models.autoencoder import Autoencoder, HigherOrderContractiveAutoencoder
from pylearn2.corruption import BinomialCorruptor
from pylearn2.config import yaml_parse
from theano.tensor.basic import _allclose

def test_autoencoder_logistic_linear_tied():
    data = np.random.randn(10, 5).astype(config.floatX)
    ae = Autoencoder(5, 7, act_enc='sigmoid', act_dec='linear',
                     tied_weights=True)
    w = ae.weights.get_value()
    ae.hidbias.set_value(np.random.randn(7).astype(config.floatX))
    hb = ae.hidbias.get_value()
    ae.visbias.set_value(np.random.randn(5).astype(config.floatX))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.dot(1. / (1 + np.exp(-hb - np.dot(data,  w))), w.T) + vb
    ff = theano.function([d], ae.reconstruct(d))
    assert _allclose(ff(data), result)


def test_autoencoder_tanh_cos_untied():
    data = np.random.randn(10, 5).astype(config.floatX)
    ae = Autoencoder(5, 7, act_enc='tanh', act_dec='cos',
                     tied_weights=False)
    w = ae.weights.get_value()
    w_prime = ae.w_prime.get_value()
    ae.hidbias.set_value(np.random.randn(7).astype(config.floatX))
    hb = ae.hidbias.get_value()
    ae.visbias.set_value(np.random.randn(5).astype(config.floatX))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.cos(np.dot(np.tanh(hb + np.dot(data,  w)), w_prime) + vb)
    ff = theano.function([d], ae.reconstruct(d))
    assert _allclose(ff(data), result)


def test_high_order_autoencoder_init():
    """
    Just test that model initialize and return
    the penalty without error.
    """
    corruptor = BinomialCorruptor(corruption_level = 0.5)
    model = HigherOrderContractiveAutoencoder(
            corruptor = corruptor,
            num_corruptions = 2,
            nvis = 5,
            nhid = 7,
            act_enc = 'sigmoid',
            act_dec = 'sigmoid')

    X = tensor.matrix()
    data = np.random.randn(10, 5).astype(config.floatX)
    ff = theano.function([X], model.higher_order_penalty(X))
    assert type(ff(data)) == np.ndarray


def test_cae_basic():
    """
    Tests that we can load a contractive autoencoder
    and train it for a few epochs (without saving) on a dummy
    dataset-- tiny model and dataset
    """

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 3, 16] },
            num_examples: 10,
            dim: 5,
            num_classes: 5
        },
        model: !obj:pylearn2.models.autoencoder.ContractiveAutoencoder {
            nvis: 5,
            nhid: 5,
            irange: 0.05,
            act_enc: "sigmoid",
            act_dec: "sigmoid"
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 10,
            learning_rate: .1,
            monitoring_dataset:
                {
                    'train' : *train
                },
            cost: !obj:pylearn2.costs.cost.SumOfCosts {
            costs: [
                !obj:pylearn2.costs.autoencoder.MeanBinaryCrossEntropy {},
                [0.1, !obj:pylearn2.costs.cost.MethodCost { method: contraction_penalty }]
            ]
        },
           termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
        },
    }
    """

    train = yaml_parse.load(yaml_string)
    train.main_loop()

def test_hcae_basic():
    """
    Tests that we can load a higher order contractive autoencoder
    and train it for a few epochs (without saving) on a dummy
    dataset-- tiny model and dataset
    """

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 3, 16] },
            num_examples: 10,
            dim: 5,
            num_classes: 5
        },
        model: !obj:pylearn2.models.autoencoder.HigherOrderContractiveAutoencoder {
            nvis: 5,
            nhid: 5,
            irange: 0.05,
            act_enc: "sigmoid",
            act_dec: "sigmoid",
            num_corruptions: 2,
            corruptor: !obj:pylearn2.corruption.BinomialCorruptor {
                corruption_level: 0.5
            }
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 10,
            learning_rate: .1,
            monitoring_dataset:
                {
                    'train' : *train
                },
            cost: !obj:pylearn2.costs.cost.SumOfCosts {
            costs: [
                !obj:pylearn2.costs.autoencoder.MeanBinaryCrossEntropy {},
                [0.1, !obj:pylearn2.costs.cost.MethodCost { method: higher_order_penalty }]
            ]
        },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
        },
    }
    """

    train = yaml_parse.load(yaml_string)
    train.main_loop()
