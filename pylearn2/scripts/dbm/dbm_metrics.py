#!/usr/bin/env python
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Guillaume Desjargins", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

"""
This script computes both an estimate of the partition function of the provided
DBM model and an estimate of the log-likelihood on the given training and test
sets.

This is guaranteed to work only for DBMs with a BinaryVector visible layer and
BinaryVectorMaxPool hidden layers with pool sizes of 1.

It uses annealed importance sampling (AIS) to estimate Z, the partition
function.

TODO: add more details, cite paper


usage: dbm_metrics.py [-h] {ais} model_path

positional arguments:
    {ais}       the desired metric
    model_path  path to the pickled DBM model

optional arguments:
    -h, --help  show the help message and exit
"""

import argparse
import warnings
import numpy
import logging

from theano.compat.six.moves import xrange
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import scan

import pylearn2
from pylearn2.compat import OrderedDict
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from pylearn2 import utils

floatX = theano.config.floatX
logging.basicConfig(level=logging.INFO)
rng = numpy.random.RandomState(9873242)
theano_rng = RandomStreams(rng.randint(2**30))


def _sample_even_odd(W_list, b_list, samples, beta, odd=True):
    """
    Sample from the even (or odd) layers given a list of previous states.

    Parameters
    ----------
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    samples : array-like object of theano shared variables
        Samples corresponding to the previous states
    beta : theano.tensor.scalar
        Inverse temperature parameter
    odd : boolean
        Whether to sample from the odd or the even layers (defaults to sampling
        from odd layers)
    """
    for i in xrange(odd, len(samples), 2):
        samples[i] = sample_hi_given(samples, i, W_list, b_list, beta)


def _activation_even_odd(W_list, b_list, samples, beta, odd=True):
    """
    Compute the activation of the even (or odd) layers given a list of
    previous states.

    Parameters
    ----------
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    samples : array-like object of theano shared variables
        Samples corresponding to the previous states
    beta : theano.tensor.scalar
        Inverse temperature parameter
    odd : boolean
        Whether to compute activation for the odd or the even layers (defaults
        to computing for odd layers)
    """
    for i in xrange(odd, len(samples), 2):
        samples[i] = hi_given(samples, i, W_list, b_list, beta,
                              apply_sigmoid=False)


def neg_sampling(W_list, b_list, nsamples, beta=1.0, pa_bias=None,
                 marginalize_odd=True, theano_rng=None):
    """
    Generate a sample from the intermediate distribution defined at inverse
    temperature 'beta', starting from state 'nsamples'. See file docstring for
    equation of p_k(h1).

    Parameters
    ----------
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    nsamples : array-like object of theano shared variables
        Negative samples corresponding to the previous states
    beta : theano.tensor.scalar
        Inverse temperature parameter
    marginalize_odd : boolean
        Whether to marginalize odd layers
    theano_rng : theano RandomStreams
        Random number generator

    Returns
    -------
    new_nsamples : array-like object of symbolic matrices
        new_nsamples[i] contains new samples for i-th layer.
    """
    # There's as much layers in the DBM as there are bias vectors
    depth = len(b_list)

    new_nsamples = [nsamples[i] for i in xrange(depth)]

    # Contribution from model B, at temperature beta_k
    _sample_even_odd(W_list, b_list, new_nsamples, beta, odd=marginalize_odd)
    _activation_even_odd(W_list, b_list, new_nsamples, beta,
                         odd=not marginalize_odd)

    # Contribution from model A, at temperature (1 - beta_k)
    new_nsamples[not marginalize_odd] += pa_bias * (1. - beta)

    # Loop over all layers (not being marginalized)
    for i in xrange(not marginalize_odd, depth, 2):
        new_nsamples[i] = T.nnet.sigmoid(new_nsamples[i])
        new_nsamples[i] = theano_rng.binomial(
            size=nsamples[i].get_value().shape, n=1, p=new_nsamples[i],
            dtype=floatX
        )

    return new_nsamples


def free_energy_at_beta(W_list, b_list, samples, beta, pa_bias=None,
                        marginalize_odd=True):
    """
    Compute the free-energy of the sample 'h1_sample', for model p_k(h1).

    Parameters
    ----------
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    samples : array-like object of theano shared variable
        Samples from which we extract the samples of layer h1
    beta : theano.tensor.scalar
        Inverse temperature beta_k of model p_k(h1) at which to measure the
        free-energy.
    pa_bias : array-like object of theano shared variables
        Biases for the A model
    marginalize_odd : boolean
        Whether to marginalize odd layers

    Returns
    -------
    fe : symbolic variable
        Free-energy of sample 'h1_sample', at inverse temperature beta
    """
    # There's as much layers in the DBM as there are bias vectors
    depth = len(b_list)

    fe = 0.

    # Contribution of biases
    keep_idx = numpy.arange(not marginalize_odd, depth, 2)
    for i in keep_idx:
        fe -= T.dot(samples[i], b_list[i]) * beta

    # Contribution of biases
    marg_idx = numpy.arange(marginalize_odd, depth, 2)
    for i in marg_idx:
        from_im1 = T.dot(samples[i-1], W_list[i]) if i >= 1 else 0.
        from_ip1 = T.dot(samples[i+1], W_list[i+1].T) if i < depth-1 else 0
        net_input = (from_im1 + from_ip1 + b_list[i]) * beta
        fe -= T.sum(T.nnet.softplus(net_input), axis=1)

    fe -= T.dot(samples[not marginalize_odd], pa_bias) * (1. - beta)

    return fe


def compute_log_ais_weights(batch_size, free_energy_fn, sample_fn, betas):
    """
    Compute log of the AIS weights

    Parameters
    ----------
    batch_size : scalar
        Size of a batch of samples
    free_energy_fn : theano.function
        Function which, given temperature beta_k, computes the free energy
        of the samples stored in model.samples. This function should return
        a symbolic vector.
    sample_fn : theano.function
        Function which, given temperature beta_k, generates samples h1 ~
        p_k(h1).
    betas : array-like object of scalars
        Inverse temperature parameters for which to compute the log_ais weights

    Returns
    -------
    log_ais_w : theano.tensor.vector
        Vector containing log ais-weights
    """
    # Initialize log-ais weights
    log_ais_w = numpy.zeros(batch_size, dtype=floatX)

    # Iterate from inverse  temperature beta_k=0 to beta_k=1...
    for i in range(len(betas) - 1):
        bp, bp1 = betas[i], betas[i+1]
        log_ais_w += free_energy_fn(bp) - free_energy_fn(bp1)
        sample_fn(bp1)
        if i % 1e3 == 0:
            logging.info('Temperature %f ' % bp1)

    return log_ais_w


def estimate_from_weights(log_ais_w):
    """
    Safely compute the log-average of the ais-weights

    Parameters
    ----------
    log_ais_w : theano.tensor.vector
        Symbolic vector containing log_ais_w^{(m)}.

    Returns
    -------
    dlogz : theano.tensor.scalar
        log(Z_B) - log(Z_A)
    var_dlogz : theano.tensor.scalar
        Variance of our estimator
    """
    # Utility function for safely computing log-mean of the ais weights
    ais_w = T.vector()
    max_ais_w = T.max(ais_w)
    dlogz = T.log(T.mean(T.exp(ais_w - max_ais_w))) + max_ais_w
    log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

    # Estimate the log-mean of the AIS weights
    dlogz = log_mean(log_ais_w)

    # Estimate log-variance of the AIS weights
    # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
    m = numpy.max(log_ais_w)
    var_dlogz = (log_ais_w.shape[0] *
                 numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                 numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

    return dlogz, var_dlogz


def compute_log_za(b_list, pa_bias, marginalize_odd=True):
    """
    Compute the exact partition function of model p_A(h1)

    Parameters
    ----------
    b_list : array-like object of theano shared variables
        Biases of the DBM
    pa_bias : array-like object of theano shared variables
        Biases for the A model
    marginalize_odd : boolean
        Whether to marginalize odd layers

    Returns
    -------
    log_za : scalar
        Partition function of model A
    """
    log_za = 0.

    for i, b in enumerate(b_list):
        if i == (not marginalize_odd):
            log_za += numpy.sum(numpy.log(1 + numpy.exp(pa_bias)))
        else:
            log_za += numpy.log(2) * b.get_value().shape[0]

    return log_za


def compute_likelihood_given_logz(nsamples, psamples, batch_size, energy_fn,
                                  inference_fn, log_z, test_x):
    """
    Compute test set likelihood as below, where q is the variational
    approximation to the posterior p(h1,h2|v).

        ln p(v) \approx \sum_h q(h) E(v,h1,h2) + H(q) - ln Z

    See section 3.2 of DBM paper for details.

    Parameters
    ----------
    nsamples : array-like object of theano shared variables
        Negative samples
    psamples : array-like object of theano shared variables
        Positive samples
    batch_size : scalar
        Size of a batch of samples
    energy_fn : theano.function
        Function which computes the (temperature 1) energy of the samples. This
        function should return a symbolic vector.
    inference_fn : theano.function
        Inference function for DBM. Function takes a T.matrix as input (data)
        and returns a list of length 'length(b_list)', where the i-th element
        is an ndarray containing approximate samples of layer i.
    log_z : scalar
        Estimate partition function of 'model'.
    test_x : numpy.ndarray
        Test set data, in dense design matrix format.

    Returns
    -------
    likelihood : scalar
        Negative log-likelihood of test data under the model
    """
    i = 0.
    likelihood = 0

    for i in xrange(0, len(test_x), batch_size):

        # Recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:numpy.minimum(test_x.shape[0], i + batch_size), :], dtype=floatX)
        batch_size0 = len(x)
        if len(x) < batch_size:
            # concatenate x to have some dummy entries
            x = numpy.concatenate((x, numpy.zeros((batch_size-len(x),x.shape[1]), dtype=floatX)), axis=0)

        # Perform inference
        inference_fn(x)

        # Entropy of h(q) adds contribution to variational lower-bound
        hq = 0
        for psample in psamples[1:]:
            temp = \
                - psample.get_value() * numpy.log(1e-5 + psample.get_value()) \
                - (1.-psample.get_value()) \
                * numpy.log(1. - psample.get_value() + 1e-5)
            hq += numpy.sum(temp, axis=1)

        # Copy into negative phase buffers to measure energy
        nsamples[0].set_value(x)
        for ii, psample in enumerate(psamples):
            if ii > 0:
                nsamples[ii].set_value(psample.get_value())

        # Compute sum of likelihood for current buffer
        x_likelihood = numpy.sum((-energy_fn(1.0) + hq - log_z)[:batch_size0])

        # Perform moving average of negative likelihood
        # Divide by len(x) and not bufsize, since last buffer might be smaller
        likelihood = (i * likelihood + x_likelihood) / (i + batch_size0)

    return likelihood


def hi_given(samples, i, W_list, b_list, beta=1.0, apply_sigmoid=True):
    """
    Compute the state of hidden layer i given all other layers

    Parameters
    ----------
    samples : array-like object of theano shared variables
        For the positive phase, samples[0] points to the input, while
        samples[i] contains the current state of the i-th layer. In the
        negative phase, samples[i] contains the persistent chain associated
        with the i-th layer.
    i : integer
        Compute activation of layer i of our DBM
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    beta : scalar
        Inverse temperature parameter used when performing AIS
    apply_sigmoid : boolean
        When False, hi_given will not apply the sigmoid. Useful for AIS
        estimate.

    Returns
    -------
    hi_mean : symbolic variable
        Activation of the i-th layer
    """
    # There's as much layers in the DBM as there are bias vectors
    depth = len(samples)

    hi_mean = 0.
    if i < depth-1:
        # Top-down input
        wip1 = W_list[i+1]
        hi_mean += T.dot(samples[i+1], wip1.T) * beta

    if i > 0:
        # Bottom-up input
        wi = W_list[i]
        hi_mean += T.dot(samples[i-1], wi) * beta

    hi_mean += b_list[i] * beta

    if apply_sigmoid:
        return T.nnet.sigmoid(hi_mean)
    else:
        return hi_mean


def sample_hi_given(samples, i, W_list, b_list, beta=1.0):
    """
    Given current state of our DBM ('samples'), sample the values taken by
    the i-th layer.

    Parameters
    ----------
    samples : array-like object of theano shared variables
        For the positive phase, samples[0] points to the input, while
        samples[i] contains the current state of the i-th layer. In the
        negative phase, samples[i] contains the persistent chain associated
        with the i-th layer.
    i : integer
        Compute activation of layer i of our DBM
    W_list : array-like object of theano shared variables
        Weight matrices of the DBM. Its first element is ignored, since in the
        Pylearn2 framework a visible layer does not have a weight matrix.
    b_list : array-like object of theano shared variables
        Biases of the DBM
    beta : scalar
        Inverse temperature parameter used when performing AIS

    Returns
    -------
    hi_sample : symbolic variable
        State of the i-th layer
    """
    hi_mean = hi_given(samples, i, W_list, b_list, beta)

    hi_sample = theano_rng.binomial(
        size=samples[i].get_value().shape,
        n=1, p=hi_mean,
        dtype=floatX
    )

    return hi_sample


def _e_step(psamples, W_list, b_list, n_steps=100, eps=1e-5):
    """
    Performs 'n_steps' of mean-field inference (used to compute positive phase
    statistics)

    Parameters
    ----------
    psamples : array-like object of theano shared variables
        State of each layer of the DBM (during the inference process).
        psamples[0] points to the input
    n_steps :  integer
        Number of iterations of mean-field to perform
    """
    depth = len(psamples)

    # now alternate mean-field inference for even/odd layers
    def mf_iteration(*psamples):
        new_psamples = [p for p in psamples]
        for i in xrange(1, depth, 2):
            new_psamples[i] = hi_given(psamples, i, W_list, b_list)
        for i in xrange(2, depth, 2):
            new_psamples[i] = hi_given(psamples, i, W_list, b_list)

        score = 0.
        for i in xrange(1, depth):
            score = T.maximum(T.mean(abs(new_psamples[i] - psamples[i])),
                              score)

        return new_psamples, theano.scan_module.until(score < eps)

    new_psamples, updates = scan(
        mf_iteration,
        outputs_info=psamples,
        n_steps=n_steps
    )

    return [x[-1] for x in new_psamples]


def estimate_likelihood(W_list, b_list, trainset, testset, free_energy_fn=None,
                        batch_size=100, large_ais=False, log_z=None,
                        pos_mf_steps=50, pos_sample_steps=0):
    """
    Compute estimate of log-partition function and likelihood of trainset and
    testset

    Parameters
    ----------
    W_list : array-like object of theano shared variables
    b_list : array-like object of theano shared variables
        Biases of the DBM
    trainset : pylearn2.datasets.dataset.Dataset
        Training set
    testset : pylearn2.datasets.dataset.Dataset
        Test set
    free_energy_fn : theano.function
        Function which, given temperature beta_k, computes the free energy
        of the samples stored in model.samples. This function should return
        a symbolic vector.
    batch_size : integer
        Size of a batch of examples
    large_ais : boolean
        If True, will use 3e5 chains, instead of 3e4
    log_z : log-partition function (if precomputed)
    pos_mf_steps: the number of fixed-point iterations for approximate inference
    pos_sample_steps: same thing as pos_mf_steps
        when both pos_mf_steps > 0 and pos_sample_steps > 0,
        pos_mf_steps has a priority

    Returns
    -------
    nll : scalar
        Negative log-likelihood of data.X under `model`.
    logz : scalar
        Estimate of log-partition function of `model`.
    """

    warnings.warn("This is garanteed to work only for DBMs with a " +
                  "BinaryVector visible layer and BinaryVectorMaxPool " +
                  "hidden layers with pool sizes of 1.")

    # Add a dummy placeholder for visible layer's weights in W_list
    W_list = [None] + W_list

    # Depth of the DBM
    depth = len(b_list)

    # Initialize samples
    psamples = []
    nsamples = []
    for i, b in enumerate(b_list):
        psamples += [utils.sharedX(rng.rand(batch_size,
                                            b.get_value().shape[0]),
                                   name='psamples%i' % i)]
        nsamples += [utils.sharedX(rng.rand(batch_size,
                                            b.get_value().shape[0]),
                                   name='nsamples%i' % i)]
    psamples[0] = T.matrix('psamples0')

    ##########################
    ## BUILD THEANO FUNCTIONS
    ##########################
    beta = T.scalar()

    # For an even number of layers, we marginalize the odd layers
    # (and vice-versa)
    marginalize_odd = (depth % 2) == 0

    # Build function to retrieve energy.
    E = -T.dot(nsamples[0], b_list[0]) * beta
    for i in xrange(1, depth):
        E -= T.sum(T.dot(nsamples[i-1], W_list[i] * beta) * nsamples[i],
                   axis=1)
        E -= T.dot(nsamples[i], b_list[i] * beta)
    energy_fn = theano.function([beta], E)

    # Build inference function.
    assert (pos_mf_steps or pos_sample_steps)
    pos_steps = pos_mf_steps if pos_mf_steps else pos_sample_steps
    new_psamples = _e_step(psamples, W_list, b_list, n_steps=pos_steps)
    ups = OrderedDict()
    for psample, new_psample in zip(psamples[1:], new_psamples[1:]):
        ups[psample] = new_psample
    temp = numpy.asarray(trainset.X, dtype=floatX)
    mean_train = numpy.mean(temp, axis=0)
    inference_fn = theano.function(inputs=[psamples[0]], outputs=[],
                                   updates=ups)

    # Configure baserate bias for (h0 if `marginalize_odd` else h1)
    inference_fn(numpy.tile(mean_train, (batch_size, 1)))
    numpy_psamples = [mean_train[None, :]] + \
                     [psample.get_value() for psample in psamples[1:]]
    mean_pos = numpy.minimum(numpy_psamples[not marginalize_odd], 1-1e-5)
    mean_pos = numpy.maximum(mean_pos, 1e-5)
    pa_bias = -numpy.log(1./mean_pos[0] - 1.)

    # Build Theano function to sample from interpolating distributions.
    updates = OrderedDict()
    new_nsamples = neg_sampling(W_list, b_list, nsamples, beta=beta,
                                pa_bias=pa_bias,
                                marginalize_odd=marginalize_odd,
                                theano_rng=theano_rng)
    for (nsample, new_nsample) in zip(nsamples, new_nsamples):
        updates[nsample] = new_nsample
    sample_fn = theano.function([beta], [], updates=updates,
                                name='sample_func')

    # Build function to compute free-energy of p_k(h1).
    fe_bp_h1 = free_energy_at_beta(W_list, b_list, nsamples, beta,
                                   pa_bias, marginalize_odd=marginalize_odd)
    free_energy_fn = theano.function([beta], fe_bp_h1)

    ###########
    ## RUN AIS
    ###########

    # Generate exact sample for the base model.
    for i, nsample_i in enumerate(nsamples):
        bias = pa_bias if i == 1 else b_list[i].get_value()
        hi_mean_vec = 1. / (1. + numpy.exp(-bias))
        hi_mean = numpy.tile(hi_mean_vec, (batch_size, 1))
        r = rng.random_sample(hi_mean.shape)
        hi_sample = numpy.array(hi_mean > r, dtype=floatX)
        nsample_i.set_value(hi_sample)

    # Default configuration for interpolating distributions
    if large_ais:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e5+1)[:-1],
                         numpy.linspace(0.5, 0.9, 1e5+1)[:-1],
                         numpy.linspace(0.9, 1.0, 1e5))))
    else:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e4+1)[:-1],
                         numpy.linspace(0.5, 0.9, 1e4+1)[:-1],
                         numpy.linspace(0.9, 1.0, 1e4))))

    if log_z is None:
        log_ais_w = compute_log_ais_weights(batch_size, free_energy_fn,
                                            sample_fn, betas)
        dlogz, var_dlogz = estimate_from_weights(log_ais_w)
        log_za = compute_log_za(b_list, pa_bias, marginalize_odd)
        log_z = log_za + dlogz
        logging.info('log_z = %f' % log_z)
        logging.info('log_za = %f' % log_za)
        logging.info('dlogz = %f' % dlogz)
        logging.info('var_dlogz = %f' % var_dlogz)

    train_ll = compute_likelihood_given_logz(nsamples, psamples, batch_size,
                                             energy_fn, inference_fn, log_z,
                                             trainset.X)
    logging.info('Training likelihood = %f' % train_ll)
    test_ll = compute_likelihood_given_logz(nsamples, psamples, batch_size,
                                            energy_fn, inference_fn, log_z,
                                            testset.X)
    logging.info('Test likelihood = %f' % test_ll)

    return (train_ll, test_ll, log_z)


if __name__ == '__main__':
    # Possible metrics
    metrics = {'ais': estimate_likelihood}
    datasets = {'mnist': MNIST}

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="the desired metric",
                        choices=metrics.keys())
    parser.add_argument("dataset", help="the dataset used for computing the " +
                        "metric", choices=datasets.keys())
    parser.add_argument("model_path", help="path to the pickled DBM model")
    args = parser.parse_args()

    metric = metrics[args.metric]
    dataset = datasets[args.dataset]

    model = serial.load(args.model_path)
    layers = [model.visible_layer] + model.hidden_layers
    W_list = [theano.shared(hidden_layer.get_weights())
              for hidden_layer in model.hidden_layers]
    b_list = [theano.shared(layer.get_biases()) for layer in layers]

    trainset = dataset(which_set='train')
    testset = dataset(which_set='test')

    metric(W_list, b_list, trainset, testset, pos_mf_steps=5)
