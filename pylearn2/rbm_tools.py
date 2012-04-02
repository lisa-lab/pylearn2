import numpy
import theano
from theano import tensor, config
from theano.tensor import nnet
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def compute_log_z(rbm, free_energy_fn, max_bits=15):
    """
    Compute the log partition function of an (binary-binary) RBM.

    Parameters
    ----------
    rbm : object
        An RBM object from `pylearn2.models`.
    free_energy_fn : callable
        A callable object (e.g. Theano function) that computes the
        free energy of a stack of configuration for this RBM.
    max_bits : int
        The (base-2) log of the number of states to enumerate (and
        compute free energy for) at a time.

    Notes
    -----
    This function enumerates a sum with exponentially many terms, and
    should not be used with more than a small, toy model.
    """
    # Pick whether to iterate over visible or hidden states.
    if rbm.nvis < rbm.nhid:
        width = rbm.nvis
        type = 'vis'
    else:
        width = rbm.nhid
        type = 'hid'

    # Determine in how many steps to compute Z.
    block_bits = width if (not max_bits or width < max_bits) else max_bits
    block_size = 2 ** block_bits

    # Allocate storage for 2**block_bits of the 2**width possible
    # configurations.
    try:
        logz_data_c = numpy.zeros(
            (block_size, width),
            order='C',
            dtype=config.floatX
        )
    except MemoryError:
        raise MemoryError("failed to allocate (%d, %d) matrix of "
                          "type %s in compute_log_z; try a smaller "
                          "value of max_bits" %
                          (block_size, width, str(config.floatX)))

    # fill in the first block_bits, which will remain fixed for all
    # 2**width configs
    tensor_10D_idx = numpy.ndindex(*([2] * block_bits))
    for i, j in enumerate(tensor_10D_idx):
        logz_data_c[i, -block_bits:] = j
    try:
        logz_data = numpy.array(logz_data_c, order='F', dtype=config.floatX)
    except MemoryError:
        raise MemoryError("failed to allocate (%d, %d) matrix of "
                          "type %s in compute_log_z; try a smaller "
                          "value of max_bits" %
                          (block_size, width, str(config.floatX)))

    # Allocate storage for free-energy of all 2**width configurations.
    try:
        FE = numpy.zeros(2 ** width, dtype=config.floatX)
    except MemoryError:
        raise MemoryError("failed to allocate free energy storage array "
                          "in compute_log_z; your model is too big to use "
                          "with this function")

    # now loop 2**(width - block_bits) times, filling in the
    # most-significant bits
    for bi, up_bits in enumerate(numpy.ndindex(*([2] * (width - block_bits)))):
        logz_data[:, :width - block_bits] = up_bits
        FE[bi * block_size:(bi + 1) * block_size] = free_energy_fn(logz_data)

    alpha = numpy.max(-FE)
    log_z = numpy.log(numpy.sum(numpy.exp(-FE - alpha))) + alpha

    return log_z


def compute_nll(rbm, data, log_z, free_energy_fn, bufsize=1000, preproc=None):
    """
    TODO: document me.
    """
    i = 0.
    nll = 0
    for i in xrange(0, len(data), bufsize):
        # recast data as floatX and apply preprocessing if required
        x = numpy.array(data[i:i + bufsize, :], dtype=config.floatX)
        if preproc:
            x = preproc(x)
        # compute sum of likelihood for current buffer
        x_nll = -numpy.sum(-free_energy_fn(x) - log_z)
        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        nll = (i * nll + x_nll) / (i + len(x))
    return nll


def rbm_ais(rbm_params, n_runs, visbias_a=None, data=None,
            betas=None, key_betas=None, rng=None, seed=23098):
    """
    Implements Annealed Importance Sampling for Binary-Binary RBMs, as
    described in:
    * Neal, R. M. (1998) ``Annealed importance sampling'', Technical Report No.
      9805 (revised), Dept. of Statistics, University of Toronto, 25 pages

    * Ruslan Salakhutdinov, Iain Murray. "On the quantitative analysis of deep
      belief networks".
      Proceedings of the 25th International Conference on Machine Learning,
      p.872-879, July 5--9, 2008, Helsinki, Finland

    Parameters
    ----------
    rbm_params: list
        list of numpy.ndarrays containing model parameters:
        [weights,visbias,hidbias]

    n_runs: int
        number of particles to use in AIS simulation (size of minibatch)

    visbias_a: numpy.ndarray
        optional override for visible biases. If both visbias_a and data
        are None, visible biases will be set to the same values of the
        temperature 1 model. For best results, use the `data` parameter.

    data: matrix, numpy.ndarray
        training data used to initialize the visible biases of the base-rate
        model (usually infinite temperature), to the log-mean of the
        distribution (maximum likelihood solution assuming a zero-weight
        matrix). This ensures that the base-rate model is the "closest" to the
        model at temperature 1.

    betas: numpy.ndarray
        vector specifying inverse temperature of intermediate distributions (in
        increasing order). If None, defaults to AIS.dflt_betas

    key_betas: numpy.ndarray
        if not None, AIS.run will save the log AIS weights for all temperatures
        in `key_betas`.  This allows the user to estimate logZ at several
        temperatures in a single pass of AIS.

    rng: None or RandomStream
        Random number generator object to use.

    seed: int
        if rng is None, initialize rng with this seed.
    """
    (weights, visbias, hidbias) = rbm_params

    if rng is None:
        rng = numpy.random.RandomState(seed)

    if data is None:
        if visbias_a is None:
            # configure base-rate biases to those supplied by user
            visbias_a = visbias
        else:
            visbias_a = visbias_a
    else:
        # set biases of base-rate model to ML solution
        data = numpy.asarray(data, dtype=config.floatX)
        data = numpy.mean(data, axis=0)
        data = numpy.minimum(data, 1 - 1e-5)
        data = numpy.maximum(data, 1e-5)
        visbias_a = -numpy.log(1. / data - 1)
    hidbias_a = numpy.zeros_like(hidbias)
    weights_a = numpy.zeros_like(weights)
    # generate exact sample for the base model
    v0 = numpy.tile(1. / (1 + numpy.exp(-visbias_a)), (n_runs, 1))
    v0 = numpy.array(v0 > rng.random_sample(v0.shape), dtype=config.floatX)
    # we now compute the log AIS weights for the ratio log(Zb/Za)
    ais = rbm_z_ratio((weights_a, visbias_a, hidbias_a),
                      rbm_params, n_runs, v0,
                      betas=betas, key_betas=key_betas, rng=rng)
    dlogz, var_dlogz = ais.estimate_from_weights()
    # log Z = log_za + dlogz
    ais.log_za = weights_a.shape[1] * numpy.log(2) + \
                 numpy.sum(numpy.log(1 + numpy.exp(visbias_a)))
    ais.log_zb = ais.log_za + dlogz
    return (ais.log_zb, var_dlogz), ais


def rbm_z_ratio(rbmA_params, rbmB_params, n_runs, v0=None,
                betas=None, key_betas=None, rng=None, seed=23098):
    """
    Computes the AIS log-weights log_wi, such that:
    log Zb = log Za + log 1/M \sum_{i=1}^M \exp(log_ais_wi)

    Parameters
    ----------
    rbmA_params: list
        list of numpy.ndarrays, corresponding to parameters of RBM whose
        partition Z_a is usually known (i.e. baserate model at beta=0).
        Parameters are given in the order:
        [weights, visbias, hidbias]

    rbmB_params: list
        list of numpy.ndarrays, corresponding to parameters of RBM whose
        partition Z_a is usually known (i.e. baserate model at beta=0).
        Parameters are given in the order:

    Additional parameters are as described in the docstring for
    `rbm_ais`.
    """
    # check that both models have the same number of hidden units
    assert rbmA_params[0].shape[0] == rbmB_params[0].shape[0]
    # check that both models have the same number of visible units
    assert len(rbmA_params[1]) == len(rbmB_params[1])

    if rng is None:
        rng = numpy.random.RandomState(seed)
    # make sure parameters are in floatX format for GPU support
    rbmA_params = [numpy.asarray(q, dtype=config.floatX) for q in rbmA_params]
    rbmB_params = [numpy.asarray(q, dtype=config.floatX) for q in rbmB_params]

    # declare symbolic vars for current sample `v_sample` and temp `beta`
    v_sample = tensor.matrix('ais_v_sample')
    beta = tensor.scalar('ais_beta')

    ### given current sample `v_sample`, generate new samples from inv.
    ### temperature `beta`
    new_v_sample = rbm_ais_gibbs_for_v(rbmA_params, rbmB_params,
                                       beta, v_sample)
    sample_fn = theano.function([beta, v_sample], new_v_sample)

    ### build theano function to compute the free-energy
    fe = rbm_ais_pk_free_energy(rbmA_params, rbmB_params, beta, v_sample)
    free_energy_fn = theano.function([beta, v_sample],
                                     fe, allow_input_downcast=False)

    ### RUN AIS ###
    weights_b = rbmB_params
    v0 = rng.rand(n_runs, weights_b.shape[0]) if v0 is None else v0
    ais = AIS(sample_fn, free_energy_fn, v0, n_runs)
    ais.set_betas(betas, key_betas=key_betas)
    ais.run()

    return ais


def rbm_ais_pk_free_energy(rbmA_params, rbmB_params, beta, v_sample):
    """
    Computes the free-energy of visible unit configuration `v_sample`,
    according to the interpolating distribution at temperature beta. The
    interpolating distributions are given by: p_a(v)^(1-beta) p_b(v)^beta. See
    equation 10, of Salakhutdinov & Murray 2008.

    Parameters
    ----------
    rbmA_params:
    rbmB_params:
        see rbm_z_ratio
    beta: int
        inverse temperature at which to compute the free-energy.
    v_sample: tensor.matrix
        matrix whose rows indexes into the minibatch, and columns into the data
        dimensions.

    Returns
    -------
    f : float (scalar)
       free-energy of configuration `v_sample` given by the interpolating
       distribution at temperature beta.
    """

    def rbm_fe(rbm_params, v, b):
        (weights, visbias, hidbias) = rbm_params
        vis_term = b * tensor.dot(v, visbias)
        hid_act = b * (tensor.dot(v, weights) + hidbias)
        fe = -vis_term - tensor.sum(tensor.log(1 + tensor.exp(hid_act)),
                                    axis=1)
        return fe

    fe_a = rbm_fe(rbmA_params, v_sample, (1 - beta))
    fe_b = rbm_fe(rbmB_params, v_sample, beta)
    return fe_a + fe_b


def rbm_ais_gibbs_for_v(rbmA_params, rbmB_params, beta, v_sample, seed=23098):
    """
    Parameters:
    -----------
    rbmA_params: list
        Parameters of the baserate model (usually infinite temperature). List
        should be of length 3 and contain numpy.ndarrays corresponding to model
        parameters (weights, visbias, hidbias).

    rbmB_params: list
        similar to rbmA_params, but for model at temperature 1.

    beta: theano.shared
        scalar, represents inverse temperature at which we wish to sample from.

    v_sample: theano.shared
        matrix of shape (n_runs, nvis), state of current particles.

    seed: int
        optional seed parameter for sampling from binomial units.
    """

    (weights_a, visbias_a, hidbias_a) = rbmA_params
    (weights_b, visbias_b, hidbias_b) = rbmB_params

    theano_rng = RandomStreams(seed)

    # equation 15 (Salakhutdinov & Murray 2008)
    ph_a = nnet.sigmoid((1 - beta) * (tensor.dot(v_sample, weights_a) +
                                    hidbias_a))
    ha_sample = theano_rng.binomial(size=(v_sample.shape[0], len(hidbias_a)),
                                    n=1, p=ph_a, dtype=config.floatX)

    # equation 16 (Salakhutdinov & Murray 2008)
    ph_b = nnet.sigmoid(beta * (tensor.dot(v_sample, weights_b) + hidbias_b))
    hb_sample = theano_rng.binomial(size=(v_sample.shape[0], len(hidbias_b)),
                                    n=1, p=ph_b, dtype=config.floatX)

    # equation 17 (Salakhutdinov & Murray 2008)
    pv_act = (1 - beta) * (tensor.dot(ha_sample, weights_a.T) + visbias_a) + \
                beta * (tensor.dot(hb_sample, weights_b.T) + visbias_b)
    pv = nnet.sigmoid(pv_act)
    new_v_sample = theano_rng.binomial(
        size=(v_sample.shape[0], len(visbias_b)),
        n=1, p=pv, dtype=config.floatX
    )

    return new_v_sample


class AIS(object):
    """
    Compute the log AIS weights to approximate a ratio of partition functions.

    After being initialized, the user should call the run() method which will
    compute the log AIS weights.  estimate_from_weights will then compute the
    mean & variance of log(Zb/Za) (Eq.11).

    The notation used here is slightly different than in Salakhutdinov & Murray
    2008. We write the AIS weights as follows (note that the denominator is
    always of the form p_i(x_i) to indicate that x_i ~ p_i.)

    w^i = p1(v0)*p2(v1)*...*pk(v_{k-1}) / [p0(v0)*p1(v1)*...*p_{k-1}(vk-1)]
        = p1(v0)/p0(v0) * p2(v1)/p1(v1) * ... * pk(v_{k-1})/p_{k-1}(vk-1)
    log_w^i = fe_0(v0) - fe_1(v0) +
              fe_1(v1) - fe_2(v1) + ... +
              fe_{k-1}(v_{k-1}) - fe_{k}(v_{k-1})
    """

    def fX(a):
        return numpy.asarray(a, dtype=config.floatX)

    # default configuration for interpolating distributions
    dflt_beta = numpy.hstack((fX(numpy.linspace(0, 0.5, 1e3)),
                              fX(numpy.linspace(0.5, 0.9, 1e4)),
                              fX(numpy.linspace(0.9, 1.0, 1e4))))

    def __init__(self, sample_fn, free_energy_fn, v_sample0, n_runs,
                 log_int=500):
        """
        Initialized the AIS object.

        Parameters
        ----------
        sample_fn: compiled theano function, sample_fn(beta, v_sample)
            returns new model samples, at inverse temperature `beta`.
            Internally, we do this by performing block gibbs sampling using
            Eq.(15-17) (implemented in rbm_ais_gibbs_for_v) starting from
            configuration v_sample.

        free_energy_fn: theano function, free_energy_fn(beta,v_sample)
            Computes the free-energy of of configuration v_sample at the
            interpolating distribution p_a^(1-beta) p_b^(beta).

        v_sample0: numpy.ndarray
            initial samples from model A.

        n_runs: int
            number of AIS runs (i.e. minibatch size)

        log_int: int
            log standard deviation of log ais weights every `log_int`
            temperatures.
        """

        self.sample_fn = sample_fn
        self.free_energy_fn = free_energy_fn
        self.v_sample0 = v_sample0
        self.n_runs = n_runs
        self.log_int = log_int

        # initialize log importance weights
        self.log_ais_w = numpy.zeros(n_runs, dtype=config.floatX)

        # utility function for safely computing log-mean of the ais weights
        ais_w = tensor.vector()
        dlogz = (
            tensor.log(tensor.mean(tensor.exp(ais_w - tensor.max(ais_w)))) \
                + tensor.max(ais_w)
        )
        self.log_mean = theano.function([ais_w], dlogz,
                                        allow_input_downcast=False)

    def set_betas(self, betas=None, key_betas=None):
        """
        Set the inverse temperature parameters of the AIS procedure.

        Parameters
        ----------
        betas : ndarray, optional
            Vector of temperatures specifying interpolating distributions

        key_betas : ndarray, optional
            If specified (not None), specifies specific temperatures at which
            we want to compute the AIS estimate. AIS.run will then return a
            vector, containing AIS at each key_beta temperature, including the
            nominal temperature.
        """
        self.key_betas = None if key_betas is None else numpy.sort(key_betas)

        betas = numpy.array(betas, dtype=config.floatX) \
                if betas is not None else self.dflt_beta
        # insert key temperatures within
        if key_betas is not None:
            betas = numpy.hstack((betas, key_betas))
            betas.sort()

        self.betas = betas

    def run(self, n_steps=1):
        """
        Performs the grunt-work, implementing...
        log_w^i += fe_{k-1}(v_{k-1}) - fe_{k}(v_{k-1}) recursively for all
        temperatures.
        """
        if not hasattr(self, 'betas'):
            self.set_betas()

        self.std_ais_w = []  # used to log std of log_ais_w regularly
        self.logz_beta = []  # used to log log_ais_w at every `key_beta` value
        self.var_logz_beta = []  # used to log variance of log_ais_w as above

        # initial sample
        state = self.v_sample0
        ki = 0

        # loop over all temperatures from beta=0 to beta=1
        for i in range(len(self.betas) - 1):
            bp, bp1 = self.betas[i], self.betas[i + 1]
            # log-ratio of (free) energies for two nearby temperatures
            self.log_ais_w += self.free_energy_fn(bp,  state) - \
                              self.free_energy_fn(bp1, state)
            # log standard deviation of AIS weights (kind of deprecated)
            if (i + 1) % self.log_int == 0:
                m = numpy.max(self.log_ais_w)
                std_ais = (numpy.log(numpy.std(numpy.exp(self.log_ais_w - m)))
                           + m - numpy.log(self.n_runs) / 2)
                self.std_ais_w.append(std_ais)

            # whenever we reach a "key" beta value, log log_ais_w and
            # var(log_ais_w) so we can estimate log_Z_{beta=key_betas[i]} after
            # the fact.
            if self.key_betas is not None and \
               ki < len(self.key_betas) and \
               bp1 == self.key_betas[ki]:

                log_ais_w_bi, var_log_ais_w_bi = \
                    self.estimate_from_weights(self.log_ais_w)
                self.logz_beta.insert(0, log_ais_w_bi)
                self.var_logz_beta.insert(0, var_log_ais_w_bi)
                ki += 1

            # generate a new sample at temperature beta_{i+1}
            state = self.sample_fn(bp1, state)

    def estimate_from_weights(self, log_ais_w=None):
        """
        Once run() method has been called, estimates the mean and variance of
        log(Zb/Za).

        Parameters
        ----------
        log_ais_w: None or 1D numpy.ndarray
            optional override for log_ais_w. When None, estimates log(Zb/Za)
            using the log AIS weights computed by AIS.run() method.

        Returns
        -------
        f: float
            Estimated mean of log(Zb/Za), log-ratio of partition functions of
            model B and A.

        v: float
            Estimated variance of log(Zb/Za)
        """

        log_ais_w = self.log_ais_w if log_ais_w is None else log_ais_w

        # estimate the log-mean of the AIS weights
        dlogz = self.log_mean(log_ais_w)

        # estimate log-variance of the AIS weights
        # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
        m = numpy.max(log_ais_w)
        var_dlogz = (log_ais_w.shape[0] *
                     numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                     numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

        return dlogz, var_dlogz
