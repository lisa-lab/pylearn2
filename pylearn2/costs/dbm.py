"""
This module contains cost functions to use with deep Boltzmann machines
(pylearn2.models.dbm).
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import warnings

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.models import dbm
from pylearn2.models.dbm import flatten
from pylearn2.utils import safe_zip

class PCD(Cost):
    """
    An intractable cost representing the variational upper bound
    on the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.

    TODO add citation to Tieleman paper, Younes paper
    """

    def __init__(self, num_chains, num_gibbs_steps, supervised = False):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        assert supervised in [True, False]

    def __call__(self, model, X, Y=None):
        """
        The partition function makes this intractable.
        """

        if self.supervised:
            assert Y is not None

        return None

    def get_monitoring_channels(self, model, X, Y = None):
        rval = {}

        history = model.mf(X, return_history = True)
        q = history[-1]

        if self.supervised:
            assert Y is not None
            Y_hat = q[-1]
            true = T.argmax(Y,axis=1)
            pred = T.argmax(Y_hat, axis=1)

            #true = Print('true')(true)
            #pred = Print('pred')(pred)

            wrong = T.neq(true, pred)
            err = T.cast(wrong.mean(), X.dtype)
            rval['err'] = err

        return rval



    def get_gradients(self, model, X, Y=None):
        """
        PCD approximation to the gradient of the bound.
        Keep in mind this is a cost, so we are upper bounding
        the negative log likelihood.
        """

        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive phase
            assert isinstance(model.hidden_layers[-1], dbm.Softmax)



        q = model.mf(X, Y)


        """
            Use the non-negativity of the KL divergence to construct a lower bound
            on the log likelihood. We can drop all terms that are constant with
            repsect to the model parameters:

            log P(v) = L(v, q) + KL(q || P(h|v))
            L(v, q) = log P(v) - KL(q || P(h|v))
            L(v, q) = log P(v) - sum_h q(h) log q(h) + q(h) log P(h | v)
            L(v, q) = log P(v) + sum_h q(h) log P(h | v) + const
            L(v, q) = log P(v) + sum_h q(h) log P(h, v) - sum_h q(h) log P(v) + const
            L(v, q) = sum_h q(h) log P(h, v) + const
            L(v, q) = sum_h q(h) -E(h, v) - log Z + const

            so the cost we want to minimize is
            expected_energy + log Z + const


            Note: for the RBM, this bound is exact, since the KL divergence goes to 0.
        """

        variational_params = flatten(q)

        # The gradients of the expected energy under q are easy, we can just do that in theano
        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = dict(safe_zip(params, T.grad(expected_energy_q, params,
            consider_constant = variational_params,
            disconnected_inputs = 'ignore')))

        """
        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """

        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        orig_V = layer_to_chains[model.visible_layer] # rm
        orig_Y = layer_to_chains[model.hidden_layers[-1]] # rm

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(layer_to_chains,
                self.theano_rng, num_steps=self.num_gibbs_steps,
                return_layer_to_updated = True)

        assert self.num_gibbs_steps >= len(model.hidden_layers) # rm
        assert orig_Y in theano.gof.graph.ancestors([layer_to_chains[model.visible_layer]]) # rm
        assert updates[orig_V] is layer_to_chains[model.visible_layer] #rm

        warnings.warn("""TODO: reduce variance of negative phase by integrating out
                the even-numbered layers. The Rao-Blackwellize method can do this
                for you when expected gradient = gradient of expectation, but doing
                this in general is trickier.""")
        #layer_to_chains = model.rao_blackwellize(layer_to_chains)

        expected_energy_p = model.energy(layer_to_chains[model.visible_layer],
                [layer_to_chains[layer] for layer in model.hidden_layers]).mean()

        samples = flatten(layer_to_chains.values())
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = dict(safe_zip(params, T.grad(-expected_energy_p, params, consider_constant
            = samples, disconnected_inputs='ignore')))


        for param in list(gradients.keys()):
            #print param.name,': '
            #print theano.printing.min_informative_str(neg_phase_grads[param])
            gradients[param] =  neg_phase_grads[param]  + gradients[param]

        return gradients, updates

