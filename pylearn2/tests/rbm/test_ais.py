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
    model.weights.set_value(rbm_params[0])
    model.visbias.set_value(rbm_params[1])
    model.hidbias.set_value(rbm_params[2])

    hid = T.matrix('hid')
    hid_fe = model.free_energy_given_h(hid)
    free_energy_fn = theano.function([hid], hid_fe)

    return rbm_tools.compute_log_z(model, free_energy_fn)

def ais_nodata(fname, do_exact=True):

    rbm_params = load_rbm_params(fname)

    # ais estimate using tempered models as intermediate distributions
    t1 = time.time()
    (logz, log_var_dz), aisobj = rbm_tools.rbm_ais(rbm_params, n_runs=100, seed=123)
    print 'AIS logZ         : %f' % logz
    print '    log_variance : %f' % log_var_dz
    print 'Elapsed time: ', time.time() - t1

    if do_exact:
        exact_logz = compute_logz(rbm_params)
        print 'Exact logZ = %f' % exact_logz
        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)

def ais_data(fname, do_exact=True):

    rbm_params = load_rbm_params(fname)

    # load data to set visible biases to ML solution
    from pylearn.datasets import MNIST
    dataset = MNIST.train_valid_test()
    data = numpy.asarray(dataset.train.x, dtype=config.floatX)

    # run ais using B=0 model with ML visible biases
    t1 = time.time()
    (logz, log_var_dz), aisobj = rbm_tools.rbm_ais(rbm_params, n_runs=100, seed=123, data=data)
    print 'AIS logZ         : %f' % logz
    print '    log_variance : %f' % log_var_dz
    print 'Elapsed time: ', time.time() - t1

    if do_exact:
        exact_logz = compute_logz(rbm_params)
        print 'Exact logZ = %f' % exact_logz
        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)

def test_ais():

    ais_data('mnistvh.mat', do_exact=True)

    # Estimate can be off when using the wrong base-rate model.
    # The test below reports log Z = 213.804 instead of 215.138
    #ais_nodata('mnistvh.mat', do_exact=True)
