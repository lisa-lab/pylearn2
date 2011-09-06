"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
import theano
import theano.tensor as tensor
#Doing this import here have no impact
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config

from pylearn2.autoencoder import Autoencoder
from theano.tensor.basic import _allclose #Handle correctly float32 and complex64

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
