"""
Test dbm_metrics script
"""
import numpy
import theano
from theano import tensor as T
from pylearn2.models.dbm.dbm import DBM
from pylearn2.models.dbm.layer import BinaryVector, BinaryVectorMaxPool
from pylearn2.scripts.dbm import dbm_metrics
from pylearn2 import rbm_tools
from pylearn2.datasets.mnist import MNIST
from nose.plugins.skip import SkipTest
from pylearn2.datasets.exc import NoDataPathError
from pylearn2.testing import no_debug_mode


@no_debug_mode
def test_ais():
    """
    Test ais computation by comparing the output of estimate_likelihood to
    Russ's code's output for the same parameters.
    """
    try:
        trainset = MNIST(which_set='train')
        testset = MNIST(which_set='test')
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")

    nvis = 784
    nhid = 20
    # Random initialization of RBM parameters
    numpy.random.seed(98734)
    w_hid = 10 * numpy.cast[theano.config.floatX](numpy.random.randn(nvis,
                                                                     nhid))
    b_vis = 10 * numpy.cast[theano.config.floatX](numpy.random.randn(nvis))
    b_hid = 10 * numpy.cast[theano.config.floatX](numpy.random.randn(nhid))

    # Initialization of RBM
    visible_layer = BinaryVector(nvis)
    hidden_layer = BinaryVectorMaxPool(detector_layer_dim=nhid, pool_size=1,
                                       layer_name='h', irange=0.1)
    rbm = DBM(100, visible_layer, [hidden_layer], 1)
    rbm.visible_layer.set_biases(b_vis)
    rbm.hidden_layers[0].set_weights(w_hid)
    rbm.hidden_layers[0].set_biases(b_hid)
    rbm.nvis = nvis
    rbm.nhid = nhid

    # Compute real logz and associated train_ll and test_ll using rbm_tools
    v_sample = T.matrix('v_sample')
    h_sample = T.matrix('h_sample')
    W = theano.shared(rbm.hidden_layers[0].get_weights())
    hbias = theano.shared(rbm.hidden_layers[0].get_biases())
    vbias = theano.shared(rbm.visible_layer.get_biases())

    wx_b = T.dot(v_sample, W) + hbias
    vbias_term = T.dot(v_sample, vbias)
    hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
    free_energy_v = -hidden_term - vbias_term
    free_energy_v_fn = theano.function(inputs=[v_sample],
                                       outputs=free_energy_v)

    wh_c = T.dot(h_sample, W.T) + vbias
    hbias_term = T.dot(h_sample, hbias)
    visible_term = T.sum(T.log(1 + T.exp(wh_c)), axis=1)
    free_energy_h = -visible_term - hbias_term
    free_energy_h_fn = theano.function(inputs=[h_sample],
                                       outputs=free_energy_h)

    real_logz = rbm_tools.compute_log_z(rbm, free_energy_h_fn)

    real_ais_train_ll = -rbm_tools.compute_nll(rbm,
                                               trainset.get_design_matrix(),
                                               real_logz, free_energy_v_fn)
    real_ais_test_ll = -rbm_tools.compute_nll(rbm, testset.get_design_matrix(),
                                              real_logz, free_energy_v_fn)

    # Compute train_ll, test_ll and logz using dbm_metrics
    train_ll, test_ll, logz = dbm_metrics.estimate_likelihood([W],
                                                              [vbias, hbias],
                                                              trainset,
                                                              testset,
                                                              pos_mf_steps=100)
    assert (real_logz - logz) < 2.0
    assert (real_ais_train_ll - train_ll) < 2.0
    assert (real_ais_test_ll - test_ll) < 2.0

if __name__ == '__main__':
    test_ais()
