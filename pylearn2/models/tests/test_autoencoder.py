"""
Tests for the pylearn2 autoencoder module.
"""
import os.path

import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from pylearn2.models.autoencoder import Autoencoder, \
    HigherOrderContractiveAutoencoder, DeepComposedAutoencoder, \
    UntiedAutoencoder, StackedDenoisingAutoencoder
from pylearn2.corruption import BinomialCorruptor
from pylearn2.config import yaml_parse
from theano.tensor.basic import _allclose


yaml_dir_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__))), 'config')


def test_autoencoder_properly_initialized():
    ae = Autoencoder(1, 1, 'sigmoid', 'linear')
    assert hasattr(ae, 'fn'), "Autoencoder didn't call Block.__init__"
    assert hasattr(ae, 'extensions'), "Autoencoder didn't call Model.__init__"


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
    corruptor = BinomialCorruptor(corruption_level=0.5)
    model = HigherOrderContractiveAutoencoder(
        corruptor=corruptor,
        num_corruptions=2,
        nvis=5,
        nhid=7,
        act_enc='sigmoid',
        act_dec='sigmoid')

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
    with open(os.path.join(yaml_dir_path, 'cae.yaml')) as f:
        yaml_string = f.read()
        train = yaml_parse.load(yaml_string)
        train.main_loop()


def test_hcae_basic():
    """
    Tests that we can load a higher order contractive autoencoder
    and train it for a few epochs (without saving) on a dummy
    dataset-- tiny model and dataset
    """
    with open(os.path.join(yaml_dir_path, 'hcae.yaml')) as f:
        yaml_string = f.read()
        train = yaml_parse.load(yaml_string)
        train.main_loop()


def test_untied_ae():
    """
    Tests that UntiedAutoencoder calls the Model superclass constructor
    """
    ae = Autoencoder(5, 7, act_enc='tanh', act_dec='cos',
                     tied_weights=True)
    model = UntiedAutoencoder(ae)
    model._ensure_extensions()


def test_dcae():
    """
    Tests that DeepComposedAutoencoder works correctly
    """
    ae = Autoencoder(5, 7, act_enc='tanh', act_dec='cos',
                     tied_weights=True)
    model = DeepComposedAutoencoder([ae])
    model._ensure_extensions()

    data = np.random.randn(10, 5).astype(config.floatX)
    model.perform(data)


def test_sdae():
    """
    Tests that StackedDenoisingAutoencoder works correctly
    """
    data = np.random.randn(10, 5).astype(config.floatX) * 100
    ae = Autoencoder(5, 7, act_enc='tanh', act_dec='cos',
                     tied_weights=False)
    corruptor = BinomialCorruptor(corruption_level=0.5)
    model = StackedDenoisingAutoencoder([ae], corruptor)
    model._ensure_extensions()

    w = ae.weights.get_value()
    w_prime = ae.w_prime.get_value()
    ae.hidbias.set_value(np.random.randn(7).astype(config.floatX))
    hb = ae.hidbias.get_value()
    ae.visbias.set_value(np.random.randn(5).astype(config.floatX))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.cos(np.dot(np.tanh(hb + np.dot(data,  w)), w_prime) + vb)
    ff = theano.function([d], model.reconstruct(d))
    assert not _allclose(ff(data), result)
