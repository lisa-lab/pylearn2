import numpy
import theano
from theano import tensor
#from pylearn.gd.sgd import sgd_updates
#from pylearn.algorithms.mcRBM import contrastive_cost, contrastive_grad

from base import Block, Trainer
from utils import sharedX

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Sampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.
    """
    def __init__(self, rbm, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def alloc(cls, rbm, particles, rng):
        self = cls()
        # TODO: Finish this (see James' code for inspiration)
        return self

class RBM(Block):
    """A base interface for RBMs."""
    def __init__(self, inputs, **kwargs):
        # TODO: Do we need anything else here?
        super(RBM, self).__init__(inputs, **kwargs)

    def cd_updates(self, pos_v, neg_v, lr, other_cost=0):
        """
        Get the contrastive gradients given positive and negative phase
        visible units, and do a gradient step on the parameters using
        the learning rates in `lr` (which is a list in the same order
        as self.params()).
        """
        # TODO: Adapt from James' code
        pass

    def gibbs_step_for_v(self, v, rng):
        """
        Do a round of block Gibbs sampling given visible configuration
        `v`, which could be training examples or "fantasy" particles.
        """
        # Implementation for binary-binary
        h_mean = self.mean_h_given_v(v)
        h_mean_shape = self.conf['batchsize'], self.conf['n_hid']
        h_sample = tensor.cast(rng.uniform(size=h_mean_shape) < h_mean, floatX)

    def mean_h_given_v(self, v):
        """
        Mean values of the hidden units given a visible configuration.
        Threshold this in order to sample.
        """
        pass

    def mean_v_given_h(self, h):
        """
        Mean reconstruction of the visible units given a hidden unit
        configuration.
        """
        pass

    def free_energy_given_v(self, v):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.
        """
        pass

    @classmethod
    def alloc(cls, inputs, conf, rng=None):
        if rng is None:
            rng = numpy.random.RandomState(conf['rbm_seed'])
        self = cls(inputs)
        self.conf = conf
        self.visbias = sharedX(
            numpy.zeros(conf['n_vis']),
            name='vb',
            borrow=True
        )
        self.hidbias = sharedX(
            numpy.zeros(conf['n_hid']),
            name='hb',
            borrow=True
        )
        self.weights = sharedX(
            .5 * rng.rand(conf['n_vis'], conf['n_hid']),
            name='W',
            borrow=True
        )

        return self

    def outputs(self):
        return (self.inputs)

class RBMTrainer(Trainer):
    @classmethod
    def alloc(cls, conf, rbm, sampler, visible_batch):
        # TODO: hyperparameters
        return cls(conf=conf, rbm=rbm, sampler=sampler,
                   visible_batch=visible_batch)
