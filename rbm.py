"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
import numpy
import theano
from theano import tensor
from theano.tensor import nnet

from .base import Block
from .utils import sharedX, safe_update

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


def training_updates(visible_batch, model, sampler, optimizer):
    """
    Get optimization updates from a given optimizer for a given
    (RBM-like) model with a given sampling strategy (sampler
    object).
    """
    pos_v = visible_batch
    neg_v = sampler.particles
    grads = model.ml_gradients(pos_v, neg_v)
    ups = optimizer.updates(gradients=grads)

    # Add the sampler's updates (negative phase particles, etc.).
    safe_update(ups, sampler.updates())
    return ups

class Sampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.
    """
    def __init__(self, rbm, particles, rng):
        self.__dict__.update(rbm=rbm)
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2**30))
        self.s_rng = RandomStreams(seed)
        self.particles = sharedX(particles, name='particles')

    def updates(self):
        """
        These are update formulas that deal with the Markov chain, not
        model parameters.
        """
        raise NotImplementedError()

class PersistentCDSampler(Sampler):
    """
    pcd_steps determines the number of steps we run the chain for.
    """
    def __init__(self, rbm, particles, rng, steps=1):
        super(PersistentCDSampler, self).__init__(rbm, particles, rng)
        self.steps = steps

    def updates(self):
        """
        These are update formulas that deal with the Markov chain, not
        model parameters.
        """
        steps = self.steps
        particles = self.particles
        # TODO: do this with scan?
        for i in xrange(steps):
            particles, _locals = self.rbm.gibbs_step_for_v(
                particles,
                self.s_rng
            )
        if not hasattr(self.rbm, 'h_sample'):
            self.rbm.h_sample = sharedX(numpy.zeros((0, 0)), 'h_sample')
        return {
            self.particles: particles,
            self.rbm.h_sample: _locals['h_mean']
        }

class RBM(Block):
    """
    A base interface for RBMs, implementing the binary-binary case.

    TODO: model shouldn't depend on batch_size.
    """
    def __init__(self, nvis, nhid, batch_size=10, irange=0.5, rng=None):
        if rng is None:
            rng = numpy.random.RandomState()
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )
        self.hidbias = sharedX(
            numpy.zeros(nhid),
            name='hb',
            borrow=True
        )
        self.weights = sharedX(
            (0.5 - rng.rand(nvis, nhid)) * irange,
            name='W',
            borrow=True
        )
        self.__dict__.update(batch_size=batch_size, nhid=nhid, nvis=nvis)
        self._params = [self.visbias, self.hidbias, self.weights]

    def ml_gradients(self, pos_v, neg_v):
        """
        Get the contrastive gradients given positive and negative phase
        visible units.
        """

        # taking the mean over each term independently allows for different
        # mini-batch sizes in the positive and negative phase.
        ml_cost = (self.free_energy_given_v(pos_v).mean() -
                   self.free_energy_given_v(neg_v).mean())

        grads = tensor.grad(ml_cost, self.params(),
                            consider_constant=[pos_v, neg_v])

        return grads

    def gibbs_step_for_v(self, v, rng):
        """
        Do a round of block Gibbs sampling given visible configuration
        `v`, which could be training examples or "fantasy" particles.
        """
        # For binary hidden units
        # TODO: factor further to extend to other kinds of hidden units
        #       (e.g. spike-and-slab)
        h_mean = self.mean_h_given_v(v)
        h_mean_shape = self.batch_size, self.nhid
        h_sample = tensor.cast(rng.uniform(size=h_mean_shape) < h_mean, floatX)
        v_mean_shape = self.batch_size, self.nvis
        # v_mean is always based on h_sample, not h_mean, because we don't
        # want h transmitting more than one bit of information per unit.
        v_mean = self.mean_v_given_h(h_sample)
        v_sample = self.sample_visibles([v_mean], v_mean_shape, rng)
        return v_sample, locals()

    def sample_visibles(self, params, shape, rng):
        v_mean = params[0]
        return tensor.cast(rng.uniform(size=shape) < v_mean, floatX)

    def input_to_h_from_v(self, v):
        return self.hidbias + tensor.dot(v, self.weights)

    def mean_h_given_v(self, v):
        """
        Mean values of the hidden units given a visible configuration.
        Threshold this in order to sample.
        """
        return nnet.sigmoid(self.input_to_h_from_v(v))

    def mean_v_given_h(self, h):
        """
        Mean reconstruction of the visible units given a hidden unit
        configuration.
        """
        return nnet.sigmoid(self.visbias + tensor.dot(h, self.weights.T))

    def free_energy_given_v(self, v):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.
        """
        sigmoid_arg = self.input_to_h_from_v(v)
        return -(tensor.dot(v, self.visbias) +
                 nnet.softplus(sigmoid_arg).sum(axis=1))

    def __call__(self, v):
        return self.mean_h_given_v(v)

    def reconstruction_error(self, v, rng):
        sample, _locals = self.gibbs_step_for_v(v, rng)
        return ((_locals['v_mean'] - v)**2).sum(axis=1) .mean()

class GaussianBinaryRBM(RBM):
    """
    An RBM with Gaussian visible units and binary hidden units.

    TODO: model shouldn't depend on batch_size.
    """
    def __init__(self, nvis, nhid, batch_size, irange=1e-3, rng=None,
                 mean_vis=False):
        super(GaussianBinaryRBM, self).__init__(nvis, nhid, batch_size, irange, rng)
        self.sigma = sharedX(
            numpy.ones(nvis),
            name='sigma',
            borrow=True
        )
        self.mean_vis = mean_vis

    def input_to_h_from_v(self, v):
        return self.hidbias + tensor.dot(v / self.sigma, self.weights)

    def mean_v_given_h(self, h):
        return self.visbias + self.sigma * tensor.dot(h, self.weights.T)

    def free_energy_given_v(self, v):
        hid_inp = self.input_to_h_from_v(v)
        squared_term = (self.visbias - v)**2 / self.sigma
        return squared_term.sum(axis=1) - nnet.softplus(hid_inp).sum(axis=1)

    def sample_visibles(self, params, shape, rng):
        v_mean = params[0]
        if self.mean_vis:
            return v_mean
        else:
            # zero mean, std sigma noise
            zero_mean = rng.normal(size=shape) * self.sigma
            return zero_mean + v_mean
