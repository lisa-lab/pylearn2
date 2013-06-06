#!/usr/bin/env python
import argparse
import numpy
import theano
import logging
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2 import utils
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
from theano.sandbox.scan import scan
from pylearn2.datasets.mnist import MNIST

floatX = theano.config.floatX
logging.basicConfig(level=logging.INFO)
rng = numpy.random.RandomState(9873242)
theano_rng = RandomStreams(rng.randint(2**30))


def _sample_even_odd(W_list, b_list, samples, beta, odd=True):
    for i in xrange(odd, len(samples), 2):
        samples[i] = sample_hi_given(samples, i, W_list, b_list, beta)


def _activation_even_odd(W_list, b_list, samples, beta, odd=True):
    for i in xrange(odd, len(samples), 2):
        samples[i] = hi_given(samples, i, W_list, b_list, beta,
                              apply_sigmoid=False)


def neg_sampling(W_list, b_list, nsamples, beta=1.0, pa_bias=None,
                 marginalize_odd=True, theano_rng=None):
    """
    Generate a sample from the intermediate distribution defined at inverse
    temperature `beta`, starting from state `nsamples`. See file docstring for
    equation of p_k(h1).

    Inputs
    ------
    dbm: dbm.DBM object
        DBM from which to sample.
    nsamples: array-like object of shared variables.
        nsamples[i] contains current samples from i-th layer.
    beta: scalar
        Temperature at which we should sample from the model.

    Returns
    -------
    new_nsamples: array-like object of symbolic matrices
        new_nsamples[i] contains new samples for i-th layer.
    """
    depth = len(b_list)
    new_nsamples = [nsamples[i] for i in xrange(depth)]
    ### contribution from model B, at temperature beta_k
    _sample_even_odd(W_list, b_list, new_nsamples, beta, odd=marginalize_odd)
    _activation_even_odd(W_list, b_list, new_nsamples, beta,
                         odd=not marginalize_odd)
    ### contribution from model A, at temperature (1 - beta_k)
    new_nsamples[not marginalize_odd] += pa_bias * (1. - beta)
    start_idx = not 0 if marginalize_odd else 1
    # loop over all layers (not being marginalized)
    for i in xrange(not marginalize_odd, depth, 2):
        new_nsamples[i] = T.nnet.sigmoid(new_nsamples[i])
        new_nsamples[i] = theano_rng.binomial(size=nsamples[i].get_value().shape,
                                              n=1, p=new_nsamples[i],
                                              dtype=floatX)
    return new_nsamples


def free_energy_at_beta(W_list, b_list, samples, beta, pa_bias=None,
                        marginalize_odd=True):
    """
    Computes the free-energy of the sample `h1_sample`, for model p_k(h1).

    Inputs
    ------
    h1_sample: theano.shared
        Shared variable representing a sample of layer h1.
    beta: T.scalar
        Inverse temperature beta_k of model p_k(h1) at which to measure the
        free-energy.

    Returns
    -------
    Symbolic variable, free-energy of sample `h1_sample`, at inv. temp beta.
    """
    depth = len(b_list)

    keep_idx = numpy.arange(not marginalize_odd, depth, 2)
    marg_idx = numpy.arange(marginalize_odd, depth, 2)

    # contribution of biases
    fe = 0.
    for i in keep_idx:
        fe -= T.dot(samples[i], b_list[i]) * beta
    # contribution of biases
    for i in marg_idx:
        from_im1 = T.dot(samples[i-1], W_list[i]) if i >= 1 else 0.
        from_ip1 = T.dot(samples[i+1], W_list[i+1].T) if i < depth-1 else 0
        net_input = (from_im1 + from_ip1 + b_list[i]) * beta
        fe -= T.sum(T.nnet.softplus(net_input), axis=1)

    fe -= T.dot(samples[not marginalize_odd], pa_bias) * (1. - beta)

    return fe


def compute_log_ais_weights(batch_size, free_energy_fn, sample_fn, betas):
    """
    Compute log of the AIS weights.
    TODO: remove dependency on global variable model.

    Inputs
    ------
    free_energy_fn: theano.function
        Function which, given temperature beta_k, computes the free energy
        of the samples stored in model.samples. This function should return
        a symbolic vector.
    sample_fn: theano.function
        Function which, given temperature beta_k, generates samples h1 ~
        p_k(h1). These samples are stored in model.nsamples[1].

    Returns
    -------
    log_ais_w: T.vector
        Vector containing log ais-weights.
    """
    # Initialize log-ais weights.
    log_ais_w = numpy.zeros(batch_size, dtype=floatX)
    # Iterate from inv. temperature beta_k=0 to beta_k=1...
    for i in range(len(betas) - 1):
        bp, bp1 = betas[i], betas[i+1]
        log_ais_w += free_energy_fn(bp) - free_energy_fn(bp1)
        sample_fn(bp1)
        if i % 1e3 == 0:
            logging.info('Temperature %f ' % bp1)
    return log_ais_w


def estimate_from_weights(log_ais_w):
    """ Safely computes the log-average of the ais-weights.

    Inputs
    ------
    log_ais_w: T.vector
        Symbolic vector containing log_ais_w^{(m)}.

    Returns
    -------
    dlogz: scalar
        log(Z_B) - log(Z_A).
    var_dlogz: scalar
        Variance of our estimator.
    """
    # Utility function for safely computing log-mean of the ais weights.
    ais_w = T.vector()
    max_ais_w = T.max(ais_w)
    dlogz = T.log(T.mean(T.exp(ais_w - max_ais_w))) + max_ais_w
    log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

    # estimate the log-mean of the AIS weights
    dlogz = log_mean(log_ais_w)

    # estimate log-variance of the AIS weights
    # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
    m = numpy.max(log_ais_w)
    var_dlogz = (log_ais_w.shape[0] *
                 numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                 numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

    return dlogz, var_dlogz


def compute_log_za(b_list, pa_bias, marginalize_odd=True):
    """
    Compute the exact partition function of model p_A(h1).
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

    Inputs:
    -------
    model: dbm.DBM
    energy_fn: theano.function
        Function which computes the (temperature 1) energy of the samples stored
        in model.samples. This function should return a symbolic vector.
    inference_fn: theano.function
        Inference function for DBM. Function takes a T.matrix as input (data)
        and returns a list of length `length(model.n_u)`, where the i-th element
        is an ndarray containing approximate samples of layer i.
    log_z: scalar
        Estimate partition function of `model`.
    test_x: numpy.ndarray
        Test set data, in dense design matrix format.

    Returns:
    --------
    Scalar, representing negative log-likelihood of test data under the model.
    """
    i = 0.
    likelihood = 0
    for i in xrange(0, len(test_x), batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + batch_size, :], dtype=floatX)

        # TODO: determine if relevant
        # perform inference
        # setup_pos_func(x)
        inference_fn(x)

        # entropy of h(q) adds contribution to variational lower-bound
        hq = 0
        for psample in psamples[1:]:
            temp = - psample.get_value() * numpy.log(1e-5 + psample.get_value()) \
                   - (1.-psample.get_value()) * numpy.log(1. - psample.get_value() + 1e-5)
            hq += numpy.sum(temp, axis=1)

        # copy into negative phase buffers to measure energy
        nsamples[0].set_value(x)
        for ii, psample in enumerate(psamples[1:]):
            nsamples[ii].set_value(psample.get_value())

        # compute sum of likelihood for current buffer
        x_likelihood = numpy.sum(-energy_fn(1.0) + hq - log_z)

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        likelihood = (i * likelihood + x_likelihood) / (i + len(x))

    return likelihood


def hi_given(samples, i, W_list, b_list, beta=1.0, apply_sigmoid=True):
    """
    Compute the state of hidden layer i given all other layers.
    :param samples: list of tensor-like objects. For the positive phase,
                    samples[0] is points to self.input, while samples[i]
                    contains the current state of the i-th layer. In the
                    negative phase, samples[i] contains the persistent chain
                    associated with the i-th layer.
    :param i: int. Compute activation of layer i of our DBM.
    :param beta: used when performing AIS.
    :param apply_sigmoid: when False, hi_given will not apply the sigmoid.
                          Useful for AIS estimate.
    """
    depth = len(samples)

    hi_mean = 0.
    if i < depth-1:
        # top-down input
        wip1 = W_list[i+1]
        hi_mean += T.dot(samples[i+1], wip1.T) * beta

    if i > 0:
        # bottom-up input
        wi = W_list[i]
        hi_mean += T.dot(samples[i-1], wi) * beta

    hi_mean += b_list[i] * beta

    if apply_sigmoid:
        return T.nnet.sigmoid(hi_mean)
    else:
        return hi_mean


def sample_hi_given(samples, i, W_list, b_list, beta=1.0):
    """
    Given current state of our DBM (`samples`), sample the values taken by
    the i-th layer.
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
    statistics).
    :param psamples: list of tensor-like objects, representing the state of
                     each layer of the DBM (during the inference process).
                     psamples[0] points to the input.
    :param n_steps:  number of iterations of mean-field to perform.
    """
    depth = len(psamples)

    new_psamples = [T.unbroadcast(T.shape_padleft(psample))
                    for psample in psamples]

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
        states=new_psamples,
        n_steps=n_steps
    )

    return [x[0] for x in new_psamples]


def estimate_likelihood(W_list, b_list, trainset, testset, free_energy_fn=None,
                        batch_size=100, large_ais=False, log_z=None,
                        pos_mf_steps=1, pos_sample_steps=0):
    """
    Compute estimate of log-partition function and likelihood of data.X.

    Inputs:
    -------
    model: dbm.DBM
    data: pylearn2 dataset
    large_ais: if True, will use 3e5 chains, instead of 3e4
    log_z: log-partition function (if precomputed)

    Returns:
    --------
    nll: scalar
        Negative log-likelihood of data.X under `model`.
    logz: scalar
        Estimate of log-partition function of `model`.
    """

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

    # TODO: change that
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
    inference_fn(mean_train[None, :])
    numpy_psamples = [mean_train[None, :]] + [psample.get_value() for psample in psamples[1:]]
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

    ### Build function to compute free-energy of p_k(h1).
    fe_bp_h1 = free_energy_at_beta(W_list, b_list, nsamples, beta,
                                   pa_bias, marginalize_odd=marginalize_odd)
    free_energy_fn = theano.function([beta], fe_bp_h1)

    # TODO: find if it is useful
    ### Build pos_func
    #input = T.matrix('input')
    #updates = OrderedDict()
    #updates[psamples[0]] = input
    #for i in xrange(1, depth):
    #    layer_init = 0.5 * T.ones((input.shape[0], b_list[i].shape[1]))
    #    updates[psamples[i]] = layer_init
    #pos_func = theano.function([input], [], updates=updates)

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

    # default configuration for interpolating distributions
    if large_ais:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e5),
                         numpy.linspace(0.5, 0.9, 1e5),
                         numpy.linspace(0.9, 1.0, 1e5))))
    else:
        betas = numpy.cast[floatX](
            numpy.hstack((numpy.linspace(0, 0.5, 1e4),
                         numpy.linspace(0.5, 0.9, 1e4),
                         numpy.linspace(0.9, 1.0, 1e4))))

    log_z = 346.325818
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

    train_ll = compute_likelihood_given_logz(nsamples, psamples, batch_size, energy_fn,
                                             inference_fn, log_z, trainset.X)
    logging.info('Training likelihood = %f' % train_ll)
    test_ll = compute_likelihood_given_logz(nsamples, psamples, batch_size, energy_fn,
                                            inference_fn, log_z, testset.X)
    logging.info('Test likelihood = %f' % test_ll)

    return (train_ll, test_ll, log_z)


if __name__ == '__main__':
    # Possible metrics
    metrics = {'ais': estimate_likelihood}

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="the desired metric",
                        choices=metrics.keys())
    parser.add_argument("model_path", help="path to the pickled DBM model")
    args = parser.parse_args()

    metric = metrics[args.metric]
    model = serial.load(args.model_path)
    layers = [model.visible_layer] + model.hidden_layers

    W_list = [theano.shared(hidden_layer.get_weights())
              for hidden_layer in model.hidden_layers]
    b_list = [theano.shared(layer.get_biases()) for layer in layers]
    trainset = MNIST(which_set='train')
    testset = MNIST(which_set='test')

    metric(W_list, b_list, trainset, testset)
