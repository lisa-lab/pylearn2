"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from pylearn2.autoencoder import Autoencoder, HigherOrderContractiveAutoencoder
from pylearn2.corruption import BinomialCorruptor
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
            num_corruptions = 5,
            nvis = 20,
            nhid = 30,
            act_enc = 'sigmoid',
            act_dec = 'sigmoid')

    X = tensor.matrix()
    data = np.random.randn(50, 20).astype(config.floatX)
    ff = theano.function([X], model.higher_order_penalty(X))
    assert type(ff(data)) == np.ndarray
