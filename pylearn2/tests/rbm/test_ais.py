from __future__ import print_function

import numpy
import time
import warnings
try:
    from scipy import io
except ImportError:
    warnings.warn("couldn't import scipy")

import theano
from theano import config
import theano.tensor as T

from pylearn2 import rbm_tools
from pylearn2.models import rbm


def load_rbm_params(fname):
    mnistvh = io.loadmat(fname)
    rbm_params = [numpy.asarray(mnistvh['vishid'], dtype=config.floatX),
                  numpy.asarray(mnistvh['visbiases'][0], dtype=config.floatX),
                  numpy.asarray(mnistvh['hidbiases'][0], dtype=config.floatX)]
    return rbm_params


def compute_logz(rbm_params):
    (nvis, nhid) = rbm_params[0].shape

    model = rbm.RBM(nvis, nhid)
    model.transformer.get_params()[0].set_value(rbm_params[0])
    model.bias_vis.set_value(rbm_params[1])
    model.bias_hid.set_value(rbm_params[2])

    hid = T.matrix('hid')
    hid_fe = model.free_energy_given_h(hid)
    free_energy_fn = theano.function([hid], hid_fe)

    return rbm_tools.compute_log_z(model, free_energy_fn)


def ais_nodata(fname, do_exact=True, betas=None):

    rbm_params = load_rbm_params(fname)

    # ais estimate using tempered models as intermediate distributions
    t1 = time.time()
    (logz, log_var_dz), aisobj = \
        rbm_tools.rbm_ais(rbm_params, n_runs=100, seed=123, betas=betas)
    print('AIS logZ         : %f' % logz)
    print('    log_variance : %f' % log_var_dz)
    print('Elapsed time: ', time.time() - t1)

    if do_exact:
        exact_logz = compute_logz(rbm_params)
        print('Exact logZ = %f' % exact_logz)
        # accept less than 1% error
        assert abs(exact_logz - logz) < 0.01*exact_logz


def ais_data(fname, do_exact=True, betas=None):

    rbm_params = load_rbm_params(fname)

    # load data to set visible biases to ML solution
    from pylearn2.datasets.mnist import MNIST
    dataset = MNIST(which_set='train')
    data = numpy.asarray(dataset.X, dtype=config.floatX)

    # run ais using B=0 model with ML visible biases
    t1 = time.time()
    (logz, log_var_dz), aisobj = \
        rbm_tools.rbm_ais(rbm_params, n_runs=100, seed=123, data=data,
                          betas=betas)
    print('AIS logZ         : %f' % logz)
    print('    log_variance : %f' % log_var_dz)
    print('Elapsed time: ', time.time() - t1)

    if do_exact:
        exact_logz = compute_logz(rbm_params)
        print('Exact logZ = %f' % exact_logz)
        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)


def test_ais():
    if config.mode == "DEBUG_MODE":
        betas = numpy.hstack((numpy.asarray(numpy.linspace(0, 0.5, 10),
                                            dtype=config.floatX),
                              numpy.asarray(numpy.linspace(0.5, 0.9, 10),
                                            dtype=config.floatX),
                              numpy.asarray(numpy.linspace(0.9, 1.0, 10),
                                            dtype=config.floatX)))
        do_exact = False
    else:
        betas = None
        do_exact = True

    ais_data('mnistvh.mat', do_exact=do_exact, betas=betas)

    # Estimate can be off when using the wrong base-rate model.
    ais_nodata('mnistvh.mat', do_exact=do_exact, betas=betas)
