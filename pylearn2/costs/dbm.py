"""
This module contains cost functions to use with deep Boltzmann machines
(pylearn2.models.dbm).
"""

__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import collections
import numpy as np
import logging
import operator
import warnings

from theano.compat.six.moves import reduce, xrange
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
RandomStreams = MRG_RandomStreams
from theano import tensor as T

import pylearn2
from pylearn2.compat import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import (
    FixedVarDescr, DefaultDataSpecsMixin, NullDataSpecsMixin
)
from pylearn2.models import dbm
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.models.dbm import flatten
from pylearn2.models.dbm.layer import BinaryVector
from pylearn2.models.dbm import Softmax
from pylearn2 import utils
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng


logger = logging.getLogger(__name__)


class BaseCD(Cost):
    """
    Parameters
    ----------
    num_chains : int
        The number of negative chains to use with PCD / SML.
        WRITEME : how is this meant to be used with CD? Do you just need to
        set it to be equal to the batch size? If so: TODO, get rid of this
        redundant aspect of the interface.
    num_gibbs_steps : int
        The number of Gibbs steps to use in the negative phase. (i.e., if
        you want to use CD-k or PCD-k, this is "k").
    supervised : bool
        If True, requests class labels and models the joint distrbution over
        features and labels.
    toronto_neg : bool
        If True, use a bit of mean field in the negative phase.
        Ruslan Salakhutdinov's matlab code does this.
    theano_rng : MRG_RandomStreams, optional
        If specified, uses this object to generate all random numbers.
        Otherwise, makes its own random number generator.
    """

    def __init__(self, num_chains, num_gibbs_steps, supervised=False,
                 toronto_neg=False, theano_rng=None):
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = make_theano_rng(theano_rng, 2012+10+14,
                which_method="binomial")
        assert supervised in [True, False]

    def expr(self, model, data):
        """
        .. todo::

            WRITEME

        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)

        return None

    def get_monitoring_channels(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()

        if self.supervised:
            X, Y = data
        else:
            X = data
            Y = None

        history = model.mf(X, return_history = True)
        q = history[-1]

        if self.supervised:
            assert len(data) == 2
            Y_hat = q[-1]
            true = T.argmax(Y, axis=1)
            pred = T.argmax(Y_hat, axis=1)

            #true = Print('true')(true)
            #pred = Print('pred')(pred)

            wrong = T.neq(true, pred)
            err = T.cast(wrong.mean(), X.dtype)
            rval['misclass'] = err

            if len(model.hidden_layers) > 1:
                q = model.mf(X, Y=Y)
                pen = model.hidden_layers[-2].upward_state(q[-2])
                Y_recons = model.hidden_layers[-1].mf_update(state_below=pen)
                pred = T.argmax(Y_recons, axis=1)
                wrong = T.neq(true, pred)

                rval['recons_misclass'] = T.cast(wrong.mean(), X.dtype)

        return rval

    def get_gradients(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        if self.supervised:
            X, Y = data
            assert Y is not None
        else:
            X = data
            Y = None

        pos_phase_grads, pos_updates = self._get_positive_phase(model, X, Y)

        neg_phase_grads, neg_updates = self._get_negative_phase(model, X, Y)

        updates = OrderedDict()
        for key, val in pos_updates.items():
            updates[key] = val
        for key, val in neg_updates.items():
            updates[key] = val

        gradients = OrderedDict()
        for param in list(pos_phase_grads.keys()):
            gradients[param] = neg_phase_grads[param] + pos_phase_grads[param]

        return gradients, updates

    def _get_toronto_neg(self, model, layer_to_chains):
        """
        .. todo::

            WRITEME
        """
        # Ruslan Salakhutdinov's undocumented negative phase from
        # http://www.mit.edu/~rsalakhu/code_DBM/dbm_mf.m
        # IG copied it here without fully understanding it, so it
        # only applies to exactly the same model structure as
        # in that code.

        assert isinstance(model.visible_layer, BinaryVector)
        assert isinstance(model.hidden_layers[0], BinaryVectorMaxPool)
        assert model.hidden_layers[0].pool_size == 1
        assert isinstance(model.hidden_layers[1], BinaryVectorMaxPool)
        assert model.hidden_layers[1].pool_size == 1
        assert isinstance(model.hidden_layers[2], Softmax)
        assert len(model.hidden_layers) == 3

        params = list(model.get_params())

        V_samples = layer_to_chains[model.visible_layer]
        H1_samples, H2_samples, Y_samples = [layer_to_chains[layer] for
                                             layer in model.hidden_layers]

        H1_mf = model.hidden_layers[0].mf_update(
            state_below=model.visible_layer.upward_state(V_samples),
            state_above=model.hidden_layers[1].downward_state(H2_samples),
            layer_above=model.hidden_layers[1])
        Y_mf = model.hidden_layers[2].mf_update(
            state_below=model.hidden_layers[1].upward_state(H2_samples))
        H2_mf = model.hidden_layers[1].mf_update(
            state_below=model.hidden_layers[0].upward_state(H1_mf),
            state_above=model.hidden_layers[2].downward_state(Y_mf),
            layer_above=model.hidden_layers[2])

        expected_energy_p = model.energy(
            V_samples, [H1_mf, H2_mf, Y_samples]
        ).mean()

        constants = flatten([V_samples, H1_mf, H2_mf, Y_samples])

        neg_phase_grads = OrderedDict(
            safe_zip(params, T.grad(-expected_energy_p, params,
                                    consider_constant=constants)))
        return neg_phase_grads

    def _get_standard_neg(self, model, layer_to_chains):
        """
        .. todo::

            WRITEME

        TODO:reduce variance of negative phase by
             integrating out the even-numbered layers. The
             Rao-Blackwellize method can do this for you when
             expected gradient = gradient of expectation, but
             doing this in general is trickier.
        """
        params = list(model.get_params())

        # layer_to_chains = model.rao_blackwellize(layer_to_chains)
        expected_energy_p = model.energy(
            layer_to_chains[model.visible_layer],
            [layer_to_chains[layer] for layer in model.hidden_layers]
        ).mean()

        samples = flatten(layer_to_chains.values())
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = OrderedDict(
            safe_zip(params, T.grad(-expected_energy_p, params,
                                    consider_constant=samples,
                                    disconnected_inputs='ignore'))
        )
        return neg_phase_grads

    def _get_variational_pos(self, model, X, Y):
        """
        .. todo::

            WRITEME
        """
        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive
            # phase
            assert isinstance(model.hidden_layers[-1], Softmax)

        q = model.mf(X, Y)

        """
            Use the non-negativity of the KL divergence to construct a lower
            bound on the log likelihood. We can drop all terms that are
            constant with repsect to the model parameters:

            log P(v) = L(v, q) + KL(q || P(h|v))
            L(v, q) = log P(v) - KL(q || P(h|v))
            L(v, q) = log P(v) - sum_h q(h) log q(h) + q(h) log P(h | v)
            L(v, q) = log P(v) + sum_h q(h) log P(h | v) + const
            L(v, q) = log P(v) + sum_h q(h) log P(h, v)
                               - sum_h q(h) log P(v) + const
            L(v, q) = sum_h q(h) log P(h, v) + const
            L(v, q) = sum_h q(h) -E(h, v) - log Z + const

            so the cost we want to minimize is
            expected_energy + log Z + const


            Note: for the RBM, this bound is exact, since the KL divergence
                  goes to 0.
        """

        variational_params = flatten(q)

        # The gradients of the expected energy under q are easy, we can just
        # do that in theano
        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(
            safe_zip(params, T.grad(expected_energy_q,
                                    params,
                                    consider_constant=variational_params,
                                    disconnected_inputs='ignore'))
        )
        return gradients

    def _get_sampling_pos(self, model, X, Y):
        """
        .. todo::

            WRITEME
        """
        layer_to_clamp = OrderedDict([(model.visible_layer, True)])
        layer_to_pos_samples = OrderedDict([(model.visible_layer, X)])
        if self.supervised:
            # note: if the Y layer changes to something without linear energy,
            #       we'll need to make the expected energy clamp Y in the
            #       positive phase
            assert isinstance(model.hidden_layers[-1], Softmax)
            layer_to_clamp[model.hidden_layers[-1]] = True
            layer_to_pos_samples[model.hidden_layers[-1]] = Y
            hid = model.hidden_layers[:-1]
        else:
            assert Y is None
            hid = model.hidden_layers

        for layer in hid:
            mf_state = layer.init_mf_state()

            def recurse_zeros(x):
                if isinstance(x, tuple):
                    return tuple([recurse_zeros(e) for e in x])
                return x.zeros_like()
            layer_to_pos_samples[layer] = recurse_zeros(mf_state)

        layer_to_pos_samples = model.sampling_procedure.sample(
            layer_to_state=layer_to_pos_samples,
            layer_to_clamp=layer_to_clamp,
            num_steps=self.num_gibbs_steps,
            theano_rng=self.theano_rng)

        q = [layer_to_pos_samples[layer] for layer in model.hidden_layers]

        pos_samples = flatten(q)

        # The gradients of the expected energy under q are easy, we can just
        # do that in theano
        expected_energy_q = model.energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(
            safe_zip(params, T.grad(expected_energy_q, params,
                                    consider_constant=pos_samples,
                                    disconnected_inputs='ignore'))
        )
        return gradients


class PCD(DefaultDataSpecsMixin, BaseCD):
    """
    An intractable cost representing the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.

    TODO add citation to Tieleman paper, Younes paper

    See Also
    --------
    BaseCD : The base class of this class (where the constructor
        parameters are documented)
    """

    def _get_positive_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME
        """
        return self._get_sampling_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME
        """
        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple, collections.ValuesView)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(
            layer_to_chains, self.theano_rng, num_steps=self.num_gibbs_steps,
            return_layer_to_updated=True)

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, updates


class VariationalPCD(DefaultDataSpecsMixin, BaseCD):
    """
    An intractable cost representing the variational upper bound
    on the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.

    TODO add citation to Tieleman paper, Younes paper

    See Also
    --------
    BaseCD : The base class of this class (where the constructor
        parameters are documented)
    """

    def expr(self, model, data):
        """
        .. todo::

            WRITEME

        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)
        return None

    def _get_positive_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME
        """
        return self._get_variational_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME

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

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(
            layer_to_chains,
            self.theano_rng, num_steps=self.num_gibbs_steps,
            return_layer_to_updated=True)

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, updates


class VariationalPCD_VarianceReduction(DefaultDataSpecsMixin, Cost):
    """
    Like pylearn2.costs.dbm.VariationalPCD, indeed a copy-paste of it,
    but with a variance reduction rule hard-coded for 2 binary
    hidden layers and a softmax label layer
    The variance reduction rule used here is to average together the expected
    energy you get by integrating out the odd numbered layers and the
    expected energy you get by integrating out the even numbered layers.
    This is the most "textbook correct" implementation of the negative
    phase, though not the one works the best in practice ("toronto_neg").
    """

    def __init__(self, num_chains, num_gibbs_steps, supervised = False):
        """
        """
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        assert supervised in [True, False]

    def expr(self, model, data):
        """
        The partition function makes this intractable.
        """

        if self.supervised:
            X, Y = data
            assert Y is not None

        return None

    def get_monitoring_channels(self, model, data):
        rval = OrderedDict()

        if self.supervised:
            X, Y = data
        else:
            X = data
            Y = None

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
            rval['misclass'] = err

            if len(model.hidden_layers) > 1:
                q = model.mf(X, Y = Y)
                pen = model.hidden_layers[-2].upward_state(q[-2])
                Y_recons = model.hidden_layers[-1].mf_update(state_below = pen)
                pred = T.argmax(Y_recons, axis=1)
                wrong = T.neq(true, pred)

                rval['recons_misclass'] = T.cast(wrong.mean(), X.dtype)


        return rval

    def get_gradients(self, model, data):
        """
        PCD approximation to the gradient of the bound.
        Keep in mind this is a cost, so we are upper bounding
        the negative log likelihood.
        """

        if self.supervised:
            X, Y = data
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive
            # phase
            assert isinstance(model.hidden_layers[-1], dbm.Softmax)
        else:
            X = data
            Y = None



        q = model.mf(X, Y)


        """
        Use the non-negativity of the KL divergence to construct a lower bound
        on the log likelihood. We can drop all terms that are constant with
        respect to the model parameters:

        log P(v) = L(v, q) + KL(q || P(h|v))
        L(v, q) = log P(v) - KL(q || P(h|v))
        L(v, q) = log P(v) - sum_h q(h) log q(h) + q(h) log P(h | v)
        L(v, q) = log P(v) + sum_h q(h) log P(h | v) + const
        L(v, q) = log P(v) + sum_h q(h) log P(h, v) - sum_h q(h) log P(v) + C
        L(v, q) = sum_h q(h) log P(h, v) + C
        L(v, q) = sum_h q(h) - E(h, v) - log Z + C

        so the cost we want to minimize is
        expected_energy + log Z + C


        Note: for the RBM, this bound is exact, since the KL divergence
        goes to 0.
        """

        variational_params = flatten(q)

        # The gradients of the expected energy under q are easy, we can just
        # do that in theano
        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(safe_zip(params, T.grad(expected_energy_q,
            params,
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

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(layer_to_chains,
                self.theano_rng, num_steps=self.num_gibbs_steps,
                return_layer_to_updated = True)

        # Variance reduction is hardcoded for this exact model
        assert isinstance(model.visible_layer, dbm.BinaryVector)
        assert isinstance(model.hidden_layers[0], dbm.BinaryVectorMaxPool)
        assert model.hidden_layers[0].pool_size == 1
        assert isinstance(model.hidden_layers[1], dbm.BinaryVectorMaxPool)
        assert model.hidden_layers[1].pool_size == 1
        assert isinstance(model.hidden_layers[2], dbm.Softmax)
        assert len(model.hidden_layers) == 3

        V_samples = layer_to_chains[model.visible_layer]
        H1_samples, H2_samples, Y_samples = [layer_to_chains[layer] for layer
                in model.hidden_layers]

        V_mf = model.visible_layer.inpaint_update(layer_above=\
                model.hidden_layers[0],
                state_above=\
                model.hidden_layers[0].downward_state(H1_samples))
        H1_mf = model.hidden_layers[0].mf_update(state_below=\
                model.visible_layer.upward_state(V_samples),
                state_above=model.hidden_layers[1].downward_state(H2_samples),
                layer_above=model.hidden_layers[1])
        H2_mf = model.hidden_layers[1].mf_update(state_below=\
                model.hidden_layers[0].upward_state(H1_samples),
                state_above=model.hidden_layers[2].downward_state(Y_samples),
                layer_above=model.hidden_layers[2])
        Y_mf = model.hidden_layers[2].mf_update(state_below=\
                model.hidden_layers[1].upward_state(H2_samples))

        expected_energy_p = 0.5 * model.energy(V_samples, [H1_mf,
            H2_samples, Y_mf]).mean() + \
                            0.5 * model.energy(V_mf, [H1_samples,
                                H2_mf, Y_samples]).mean()

        constants = flatten([V_samples, V_mf, H1_samples, H1_mf, H2_samples,
            H2_mf, Y_mf, Y_samples])

        neg_phase_grads = OrderedDict(safe_zip(params, T.grad(
            -expected_energy_p, params, consider_constant = constants)))


        for param in list(gradients.keys()):
            gradients[param] = neg_phase_grads[param] + gradients[param]

        return gradients, updates

class VariationalCD(DefaultDataSpecsMixin, BaseCD):
    """
    An intractable cost representing the negative log likelihood of a DBM.
    The gradient of this bound is computed using a markov chain initialized
    with the training example.

    Source: Hinton, G. Training Products of Experts by Minimizing
            Contrastive Divergence
    """

    def _get_positive_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME
        """
        return self._get_variational_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME

        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """
        layer_to_clamp = OrderedDict([(model.visible_layer, True)])

        layer_to_chains = model.make_layer_to_symbolic_state(self.num_chains,
                                                             self.theano_rng)
        # The examples are used to initialize the visible layer's chains
        layer_to_chains[model.visible_layer] = X
        # If we use supervised training, we need to make sure the targets are
        # also clamped.
        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive
            # phase
            assert isinstance(model.hidden_layers[-1], Softmax)
            layer_to_clamp[model.hidden_layers[-1]] = True
            layer_to_chains[model.hidden_layers[-1]] = Y

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        # We first initialize the chain by clamping the visible layer and the
        # target layer (if it exists)
        layer_to_chains = model.sampling_procedure.sample(
            layer_to_chains,
            self.theano_rng,
            layer_to_clamp=layer_to_clamp,
            num_steps=1
        )

        # We then do the required mcmc steps
        layer_to_chains = model.sampling_procedure.sample(
            layer_to_chains,
            self.theano_rng,
            num_steps=self.num_gibbs_steps
        )

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, OrderedDict()

class MF_L1_ActCost(DefaultDataSpecsMixin, Cost):
    """
    L1 activation cost on the mean field parameters.

    Adds a cost of:

    coeff * max( abs(mean_activation - target) - eps, 0)

    averaged over units

    for each layer.

    """

    def __init__(self, targets, coeffs, eps, supervised):
        """
        targets: a list, one element per layer, specifying the activation
                each layer should be encouraged to have
                    each element may also be a list depending on the
                    structure of the layer.
                See each layer's get_l1_act_cost for a specification of
                    what the state should be.
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer
        supervised: If true, runs mean field on both X and Y, penalizing
                the layers in between only
        """
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):

        if self.supervised:
            X, Y = data
            H_hat = model.mf(X, Y= Y)
        else:
            X = data
            H_hat = model.mf(X)

        hidden_layers = model.hidden_layers
        if self.supervised:
            hidden_layers = hidden_layers[:-1]
            H_hat = H_hat[:-1]

        layer_costs = []
        for layer, mf_state, targets, coeffs, eps in \
            safe_zip(hidden_layers, H_hat, self.targets, self.coeffs,
                    self.eps):
            cost = None
            try:
                cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
            except NotImplementedError:
                assert isinstance(coeffs, float) and coeffs == 0.
                assert cost is None # if this gets triggered, there might
                    # have been a bug, where costs from lower layers got
                    # applied to higher layers that don't implement the cost
                cost = None
            if cost is not None:
                layer_costs.append(cost)


        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [cost_ for cost_ in layer_costs if cost_ != 0.]

        if len(layer_costs) == 0:
            return T.as_tensor_variable(0.)
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MF_L1_ActCost'

        assert total_cost.ndim == 0

        return total_cost

class MF_L2_ActCost(DefaultDataSpecsMixin, Cost):
    """
    An L2 penalty on the amount that the hidden unit mean field parameters
    deviate from desired target values.

    TODO: write up parameters list
    """

    def __init__(self, targets, coeffs, supervised=False):
        targets = fix(targets)
        coeffs = fix(coeffs)

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, return_locals=False, **kwargs):
        """
        .. todo::

            WRITEME

        If returns locals is True, returns (objective, locals())
        Note that this means adding / removing / changing the value of
        local variables is an interface change.
        In particular, TorontoSparsity depends on "terms" and "H_hat"
        """
        self.get_data_specs(model)[0].validate(data)
        if self.supervised:
            (X, Y) = data
        else:
            X = data
            Y = None

        H_hat = model.mf(X, Y=Y)

        terms = []

        hidden_layers = model.hidden_layers
        #if self.supervised:
        #    hidden_layers = hidden_layers[:-1]

        for layer, mf_state, targets, coeffs in \
                safe_zip(hidden_layers, H_hat, self.targets, self.coeffs):
            try:
                cost = layer.get_l2_act_cost(mf_state, targets, coeffs)
            except NotImplementedError:
                if isinstance(coeffs, float) and coeffs == 0.:
                    cost = 0.
                else:
                    raise
            terms.append(cost)


        objective = sum(terms)

        if return_locals:
            return objective, locals()
        return objective


def fix(l):
    """
    Parameters
    ----------
    l : object

    Returns
    -------
    l : object
        If `l` is anything but a string, the return is the
        same as the input, but it may have been modified in place.
        If `l` is a string, the return value is `l` converted to a float.
        If `l` is a list, this function explores all nested lists inside
        `l` and turns all string members into floats.
    """
    if isinstance(l, list):
        return [fix(elem) for elem in l]
    if isinstance(l, str):
        return float(l)
    return l

class TorontoSparsity(Cost):
    """
    TODO: add link to Ruslan Salakhutdinov's paper that this is based on
    TODO: write up parameters list
    """
    def __init__(self, targets, coeffs, supervised=False):
        self.__dict__.update(locals())
        del self.self

        self.base_cost = MF_L2_ActCost(targets=targets,
                coeffs=coeffs, supervised=supervised)

    def expr(self, model, data, return_locals=False, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        return self.base_cost.expr(model, data, return_locals=return_locals,
                **kwargs)

    def get_gradients(self, model, data, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        obj, scratch = self.base_cost.expr(model, data, return_locals=True,
                                           **kwargs)
        if self.supervised:
            assert isinstance(data, (list, tuple))
            assert len(data) == 2
            (X, Y) = data
        else:
            X = data

        H_hat = scratch['H_hat']
        terms = scratch['terms']
        hidden_layers = scratch['hidden_layers']

        grads = OrderedDict()

        assert len(H_hat) == len(terms)
        assert len(terms) == len(hidden_layers)
        num_layers = len(hidden_layers)
        for i in xrange(num_layers):
            state = H_hat[i]
            layer = model.hidden_layers[i]
            term = terms[i]

            if term == 0.:
                continue
            else:
                logger.info('term is {0}'.format(term))

            if i == 0:
                state_below = X
                layer_below = model.visible_layer
            else:
                layer_below = model.hidden_layers[i-1]
                state_below = H_hat[i-1]
            state_below = layer_below.upward_state(state_below)

            components = flatten(state)

            real_grads = T.grad(term, components)

            fake_state = layer.linear_feed_forward_approximation(state_below)

            fake_components = flatten(fake_state)
            real_grads = OrderedDict(safe_zip(fake_components, real_grads))

            params = list(layer.get_params())
            fake_grads = pylearn2.utils.grad(
                cost=None,
                consider_constant=flatten(state_below),
                wrt=params,
                known_grads=real_grads
            )

            for param, grad in safe_zip(params, fake_grads):
                if param in grads:
                    grads[param] = grads[param] + grad
                else:
                    grads[param] = grad

        return grads, OrderedDict()

    def get_data_specs(self, model):
        """
        .. todo::

            WRITEME
        """
        return self.base_cost.get_data_specs(model)

class WeightDecay(NullDataSpecsMixin, Cost):
    """
    A Cost that applies the following cost function:

    coeff * sum(sqr(weights))
    for each set of weights.

    Parameters
    ----------
    coeffs : list
        One element per layer, specifying the coefficient
        to put on the L1 activation cost for each layer.
        Each element may in turn be a list, ie, for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        layer_costs = [ layer.get_weight_decay(coeff)
            for layer, coeff in safe_izip(model.hidden_layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'DBM_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost


class MultiPrediction(DefaultDataSpecsMixin, Cost):
    """
    If you use this class in your research work, please cite:

    Multi-prediction deep Boltzmann machines. Ian J. Goodfellow, Mehdi Mirza,
    Aaron Courville, and Yoshua Bengio. NIPS 2013.

    .. todo::

            WRITEME : parameters list
    """
    def __init__(self,
            monitor_multi_inference = False,
                    mask_gen = None,
                    noise = False,
                    both_directions = False,
                    l1_act_coeffs = None,
                    l1_act_targets = None,
                    l1_act_eps = None,
                    range_rewards = None,
                    stdev_rewards = None,
                    robustness = None,
                    supervised = False,
                    niter = None,
                    block_grad = None,
                    vis_presynaptic_cost = None,
                    hid_presynaptic_cost = None,
                    reweighted_act_coeffs = None,
                    reweighted_act_targets = None,
                    toronto_act_targets = None,
                    toronto_act_coeffs = None,
                    monitor_each_step = False,
                    use_sum = False
                    ):
        self.__dict__.update(locals())
        del self.self


    def get_monitoring_channels(self, model, data, drop_mask = None,
            drop_mask_Y = None, **kwargs):
        """
        .. todo::

            WRITEME
        """

        if self.supervised:
            X, Y = data
        else:
            X = data
            Y = None

        if self.supervised:
            assert Y is not None

        rval = OrderedDict()

        # TODO: shouldn't self() handle this?
        if drop_mask is not None and drop_mask.ndim < X.ndim:
            if self.mask_gen is not None:
                assert self.mask_gen.sync_channels
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        if Y is None:
            data = X
        else:
            data = (X, Y)
        scratch = self.expr(model, data, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y,
                return_locals = True)

        history = scratch['history']
        new_history = scratch['new_history']
        new_drop_mask = scratch['new_drop_mask']
        new_drop_mask_Y = None
        drop_mask = scratch['drop_mask']
        if self.supervised:
            drop_mask_Y = scratch['drop_mask_Y']
            new_drop_mask_Y = scratch['new_drop_mask_Y']

        ii = 0
        for name in ['inpaint_cost', 'l1_act_cost', 'toronto_act_cost',
                'reweighted_act_cost']:
            var = scratch[name]
            if var is not None:
                rval['total_inpaint_cost_term_'+str(ii)+'_'+name] = var
                ii = ii + 1

        if self.monitor_each_step:
            for ii, packed in enumerate(safe_izip(history, new_history)):
                state, new_state = packed
                rval['all_inpaint_costs_after_' + str(ii)] = \
                        self.cost_from_states(state,
                        new_state,
                        model, X, Y, drop_mask, drop_mask_Y,
                        new_drop_mask, new_drop_mask_Y)

                if ii > 0:
                    prev_state = history[ii-1]
                    V_hat = state['V_hat']
                    prev_V_hat = prev_state['V_hat']
                    rval['max_pixel_diff[%d]'%ii] = abs(V_hat-prev_V_hat).max()

        final_state = history[-1]

        #empirical beta code--should be moved to gaussian visible layer,
        #should support topo data
        #V_hat = final_state['V_hat']
        #err = X - V_hat
        #masked_err = err * drop_mask
        #sum_sqr_err = T.sqr(masked_err).sum(axis=0)
        #recons_count = T.cast(drop_mask.sum(axis=0), 'float32')

        # empirical_beta = recons_count / sum_sqr_err
        # assert empirical_beta.ndim == 1


        #rval['empirical_beta_min'] = empirical_beta.min()
        #rval['empirical_beta_mean'] = empirical_beta.mean()
        #rval['empirical_beta_max'] = empirical_beta.max()

        layers = model.get_all_layers()
        states = [ final_state['V_hat'] ] + final_state['H_hat']

        for layer, state in safe_izip(layers, states):
            d = layer.get_monitoring_channels_from_state(state)
            for key in d:
                mod_key = 'final_inpaint_' + layer.layer_name + '_' + key
                assert mod_key not in rval
                rval[mod_key] = d[key]

        if self.supervised:
            inpaint_Y_hat = history[-1]['H_hat'][-1]
            err = T.neq(T.argmax(inpaint_Y_hat, axis=1), T.argmax(Y, axis=1))
            assert err.ndim == 1
            assert drop_mask_Y.ndim == 1
            err =  T.dot(err, drop_mask_Y) / drop_mask_Y.sum()
            if err.dtype != inpaint_Y_hat.dtype:
                err = T.cast(err, inpaint_Y_hat.dtype)

            rval['inpaint_err'] = err

            Y_hat = model.mf(X)[-1]

            Y = T.argmax(Y, axis=1)
            Y = T.cast(Y, Y_hat.dtype)

            argmax = T.argmax(Y_hat,axis=1)
            if argmax.dtype != Y_hat.dtype:
                argmax = T.cast(argmax, Y_hat.dtype)
            err = T.neq(Y , argmax).mean()
            if err.dtype != Y_hat.dtype:
                err = T.cast(err, Y_hat.dtype)

            rval['err'] = err

            if self.monitor_multi_inference:
                Y_hat = model.inference_procedure.multi_infer(X)

                argmax = T.argmax(Y_hat,axis=1)
                if argmax.dtype != Y_hat.dtype:
                    argmax = T.cast(argmax, Y_hat.dtype)
                err = T.neq(Y , argmax).mean()
                if err.dtype != Y_hat.dtype:
                    err = T.cast(err, Y_hat.dtype)

                rval['multi_err'] = err

        return rval

    def expr(self, model, data, drop_mask = None, drop_mask_Y = None,
            return_locals = False, include_toronto = True, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        if self.supervised:
            X, Y = data
        else:
            X = data
            Y = None

        if not self.supervised:
            assert drop_mask_Y is None
            # ignore Y if some other cost is supervised and has made it get
            # passed in (can this still happen after the (space, source)
            # interface change?)
            Y = None
        if self.supervised:
            assert Y is not None
            if drop_mask is not None:
                assert drop_mask_Y is not None

        if not hasattr(model,'cost'):
            model.cost = self
        if not hasattr(model,'mask_gen'):
            model.mask_gen = self.mask_gen

        dbm = model

        X_space = model.get_input_space()

        if drop_mask is None:
            if self.supervised:
                drop_mask, drop_mask_Y = self.mask_gen(X, Y, X_space=X_space)
            else:
                drop_mask = self.mask_gen(X, X_space=X_space)

        if drop_mask_Y is not None:
            assert drop_mask_Y.ndim == 1

        if drop_mask.ndim < X.ndim:
            if self.mask_gen is not None:
                assert self.mask_gen.sync_channels
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        if not hasattr(self,'noise'):
            self.noise = False

        history = dbm.do_inpainting(X, Y = Y, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y, return_history = True,
                noise = self.noise,
                niter = self.niter, block_grad = self.block_grad)
        final_state = history[-1]

        new_drop_mask = None
        new_drop_mask_Y = None
        new_history = [ None for state in history ]

        if not hasattr(self, 'both_directions'):
            self.both_directions = False
        if self.both_directions:
            new_drop_mask = 1. - drop_mask
            if self.supervised:
                new_drop_mask_Y = 1. - drop_mask_Y
            new_history = dbm.do_inpainting(X, Y = Y,
                    drop_mask=new_drop_mask,
                    drop_mask_Y=new_drop_mask_Y, return_history=True,
                    noise = self.noise,
                    niter = self.niter, block_grad = self.block_grad)

        new_final_state = new_history[-1]

        total_cost, sublocals = self.cost_from_states(final_state,
                new_final_state, dbm, X, Y, drop_mask, drop_mask_Y,
                new_drop_mask, new_drop_mask_Y,
                return_locals=True)
        l1_act_cost = sublocals['l1_act_cost']
        inpaint_cost = sublocals['inpaint_cost']
        reweighted_act_cost = sublocals['reweighted_act_cost']

        if not hasattr(self, 'robustness'):
            self.robustness = None
        if self.robustness is not None:
            inpainting_H_hat = history[-1]['H_hat']
            mf_H_hat = dbm.mf(X, Y=Y)
            if self.supervised:
                inpainting_H_hat = inpainting_H_hat[:-1]
                mf_H_hat = mf_H_hat[:-1]
                for ihh, mhh in safe_izip(flatten(inpainting_H_hat),
                        flatten(mf_H_hat)):
                    total_cost += self.robustness * T.sqr(mhh-ihh).sum()

        if not hasattr(self, 'toronto_act_targets'):
            self.toronto_act_targets = None
        toronto_act_cost = None
        if self.toronto_act_targets is not None and include_toronto:
            toronto_act_cost = 0.
            H_hat = history[-1]['H_hat']
            for s, c, t in zip(H_hat, self.toronto_act_coeffs,
                    self.toronto_act_targets):
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                toronto_act_cost += c * T.sqr(m-t).mean()
            total_cost += toronto_act_cost

        if return_locals:
            return locals()

        total_cost.name = 'total_inpaint_cost'

        return total_cost

    def get_fixed_var_descr(self, model, data):
        """
        .. todo::

            WRITEME
        """
        X, Y = data

        assert Y is not None

        batch_size = model.batch_size

        drop_mask_X = sharedX(
                model.get_input_space().get_origin_batch(batch_size))
        drop_mask_X.name = 'drop_mask'

        X_space = model.get_input_space()

        updates = OrderedDict()
        rval = FixedVarDescr()
        inputs=[X, Y]

        if not self.supervised:
            update_X = self.mask_gen(X, X_space = X_space)
        else:
            drop_mask_Y = sharedX(np.ones(batch_size,))
            drop_mask_Y.name = 'drop_mask_Y'
            update_X, update_Y = self.mask_gen(X, Y, X_space)
            updates[drop_mask_Y] = update_Y
            rval.fixed_vars['drop_mask_Y'] =  drop_mask_Y
        if self.mask_gen.sync_channels:
            n = update_X.ndim
            assert n == drop_mask_X.ndim - 1
            update_X.name = 'raw_update_X'
            zeros_like_X = T.zeros_like(X)
            zeros_like_X.name = 'zeros_like_X'
            update_X = zeros_like_X + update_X.dimshuffle(0,1,2,'x')
            update_X.name = 'update_X'
        updates[drop_mask_X] = update_X

        rval.fixed_vars['drop_mask'] = drop_mask_X

        if hasattr(model.inference_procedure, 'V_dropout'):
            include_prob = model.inference_procedure.include_prob
            include_prob_V = model.inference_procedure.include_prob_V
            include_prob_Y = model.inference_procedure.include_prob_Y

            theano_rng = make_theano_rng(None, 2012+10+20,
                    which_method="binomial")
            for elem in flatten([model.inference_procedure.V_dropout]):
                updates[elem] = theano_rng.binomial(p=include_prob_V,
                        size=elem.shape, dtype=elem.dtype, n=1) / \
                                include_prob_V
            if "Softmax" in str(type(model.hidden_layers[-1])):
                hid = model.inference_procedure.H_dropout[:-1]
                y = model.inference_procedure.H_dropout[-1]
                updates[y] = theano_rng.binomial(p=include_prob_Y,
                        size=y.shape, dtype=y.dtype, n=1) / include_prob_Y
            else:
                hid = model.inference_procedure.H_dropout
            for elem in flatten(hid):
                updates[elem] =  theano_rng.binomial(p=include_prob,
                        size=elem.shape, dtype=elem.dtype, n=1) / include_prob

        rval.on_load_batch = [utils.function(inputs, updates=updates)]

        return rval

    def get_gradients(self, model, X, Y = None, **kwargs):
        """
        .. todo::

            WRITEME
        """

        if Y is None:
            data = X
        else:
            data = (X, Y)

        scratch = self.expr(model, data, include_toronto = False,
                return_locals=True, **kwargs)

        total_cost = scratch['total_cost']

        params = list(model.get_params())
        grads = dict(safe_zip(params, T.grad(total_cost, params,
            disconnected_inputs='ignore')))

        if self.toronto_act_targets is not None:
            H_hat = scratch['history'][-1]['H_hat']
            for i, packed in enumerate(safe_zip(H_hat,
                self.toronto_act_coeffs, self.toronto_act_targets)):
                s, c, t = packed
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                m_cost = c * T.sqr(m-t).mean()
                real_grads = T.grad(m_cost, s)
                if i == 0:
                    below = X
                else:
                    below = H_hat[i-1][0]
                W, = model.hidden_layers[i].transformer.get_params()
                assert W in grads
                b = model.hidden_layers[i].b

                ancestor = T.scalar()
                hack_W = W + ancestor
                hack_b = b + ancestor

                fake_s = T.dot(below, hack_W) + hack_b
                if fake_s.ndim != real_grads.ndim:
                    logger.error(fake_s.ndim)
                    logger.error(real_grads.ndim)
                    assert False
                sources = [ (fake_s, real_grads) ]

                fake_grads = T.grad(cost=None, known_grads=dict(sources),
                        wrt=[below, ancestor, hack_W, hack_b])

                grads[W] = grads[W] + fake_grads[2]
                grads[b] = grads[b] + fake_grads[3]


        return grads, OrderedDict()

    def get_inpaint_cost(self, dbm, X, V_hat_unmasked, drop_mask, state,
            Y, drop_mask_Y):
        """
        .. todo::

            WRITEME
        """
        rval = dbm.visible_layer.recons_cost(X, V_hat_unmasked, drop_mask,
                use_sum=self.use_sum)

        if self.supervised:
            # pyflakes is too dumb to see that both branches define `scale`
            scale = None
            if self.use_sum:
                scale = 1.
            else:
                scale = 1. / float(dbm.get_input_space().get_total_dimension())
            Y_hat_unmasked = state['Y_hat_unmasked']
            rval = rval + \
                    dbm.hidden_layers[-1].recons_cost(Y, Y_hat_unmasked,
                            drop_mask_Y,
                            scale)

        return rval

    def cost_from_states(self, state, new_state, dbm, X, Y, drop_mask,
            drop_mask_Y,
            new_drop_mask, new_drop_mask_Y, return_locals = False):
        """
        .. todo::

            WRITEME
        """

        if not self.supervised:
            assert drop_mask_Y is None
            assert new_drop_mask_Y is None
        if self.supervised:
            assert drop_mask_Y is not None
            if self.both_directions:
                assert new_drop_mask_Y is not None
            assert Y is not None

        V_hat_unmasked = state['V_hat_unmasked']
        assert V_hat_unmasked.ndim == X.ndim

        if not hasattr(self, 'use_sum'):
            self.use_sum = False

        inpaint_cost = self.get_inpaint_cost(dbm, X, V_hat_unmasked, drop_mask,
                state, Y, drop_mask_Y)

        if not hasattr(self, 'both_directions'):
            self.both_directions = False

        assert self.both_directions == (new_state is not None)

        if new_state is not None:

            new_V_hat_unmasked = new_state['V_hat_unmasked']

            new_inpaint_cost = dbm.visible_layer.recons_cost(X,
                    new_V_hat_unmasked, new_drop_mask)
            if self.supervised:
                new_Y_hat_unmasked = new_state['Y_hat_unmasked']
                scale = None
                raise NotImplementedError("This branch appears to be broken,"
                        "needs to define scale.")
                new_inpaint_cost = new_inpaint_cost + \
                        dbm.hidden_layers[-1].recons_cost(Y,
                                new_Y_hat_unmasked, new_drop_mask_Y, scale)
            # end if include_Y
            inpaint_cost = 0.5 * inpaint_cost + 0.5 * new_inpaint_cost
        # end if both directions

        total_cost = inpaint_cost

        if not hasattr(self, 'range_rewards'):
            self.range_rewards = None
        if self.range_rewards is not None:
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.range_rewards):
                try:
                    layer_cost = layer.get_range_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        if not hasattr(self, 'stdev_rewards'):
            self.stdev_rewards = None
        if self.stdev_rewards is not None:
            assert False # not monitored yet
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.stdev_rewards):
                try:
                    layer_cost = layer.get_stdev_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        l1_act_cost = None
        if self.l1_act_targets is not None:
            l1_act_cost = 0.
            if self.l1_act_eps is None:
                self.l1_act_eps = [ None ] * len(self.l1_act_targets)
            for layer, mf_state, targets, coeffs, eps in \
                    safe_izip(dbm.hidden_layers, state['H_hat'],
                            self.l1_act_targets, self.l1_act_coeffs,
                            self.l1_act_eps):

                assert not isinstance(targets, str)

                try:
                    layer_cost = layer.get_l1_act_cost(mf_state, targets,
                            coeffs, eps)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    l1_act_cost += layer_cost
                # end for substates
            # end for layers
            total_cost += l1_act_cost
        # end if act penalty

        if not hasattr(self, 'hid_presynaptic_cost'):
            self.hid_presynaptic_cost = None
        if self.hid_presynaptic_cost is not None:
            assert False # not monitored yet
            for c, s, in safe_izip(self.hid_presynaptic_cost, state['H_hat']):
                if c == 0.:
                    continue
                s = s[1]
                assert hasattr(s, 'owner')
                owner = s.owner
                assert owner is not None
                op = owner.op

                if not hasattr(op, 'scalar_op'):
                    raise ValueError("Expected V_hat_unmasked to be generated"
                            "by an Elemwise op, got " + str(op) + " of type "
                            + str(type(op)))
                assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
                z ,= owner.inputs

                total_cost += c * T.sqr(z).mean()

        if not hasattr(self, 'reweighted_act_targets'):
            self.reweighted_act_targets = None
        reweighted_act_cost = None
        if self.reweighted_act_targets is not None:
            reweighted_act_cost = 0.
            warnings.warn("reweighted_act_cost is hardcoded for sigmoid "
                    "layers and doesn't check that this is what we get.")
            for c, t, s in safe_izip(self.reweighted_act_coeffs,
                    self.reweighted_act_targets, state['H_hat']):
                if c == 0:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                d = T.sqr(m-t)
                weight = 1./(1e-7+s*(1-s))
                reweighted_act_cost += c * (weight * d).mean()
            total_cost += reweighted_act_cost

        total_cost.name = 'total_cost(V_hat_unmasked = %s)' % \
                V_hat_unmasked.name

        if return_locals:
            return total_cost, locals()

        return total_cost

default_seed = 20120712
class MaskGen:
    """
    A class that generates masks for multi-prediction training.

    Parameters
    ----------
    drop_prob : float
        The probability of dropping out a unit (making it a target of
        the training criterion)
    balance : bool
        WRITEME
    sync_channels : bool
        If True:
        Rather than dropping each pixel individually, drop spatial locations.
        i.e., we either drop the red, the green, and the blue pixel at (x, y),
        or we drop nothing at (x, y).
        If False:
        Drop each pixel independently.
    drop_prob_y : float, optional
        If specified, use a different drop probability for the class labels.
    seed : int
        The seed to use with MRG_RandomStreams for generating the random
        masks.
    """

    def __init__(self, drop_prob, balance = False, sync_channels = True,
            drop_prob_y = None, seed = default_seed):
        self.__dict__.update(locals())
        del self.self


    def __call__(self, X, Y = None, X_space=None):
        """
        Provides the mask for multi-prediction training. A 1 in the mask
        corresponds to a variable that should be used as an input to the
        inference process. A 0 corresponds to a variable that should be
        used as a prediction target of the multi-prediction training
        criterion.

        Parameters
        ----------
        X : Variable
            A batch of input features to mask for multi-prediction training
        Y : Variable
            A batch of input class labels to mask for multi-prediction
            Training

        Returns
        -------
        drop_mask : Variable
            A Theano expression for a random binary mask in the same shape as
            `X`
        drop_mask_Y : Variable, only returned if `Y` is not None
            A Theano expression for a random binary mask in the same shape as
            `Y`

        Notes
        -----
        Calling this repeatedly will yield the same random numbers each time.
        """
        assert X_space is not None
        self.called = True
        assert X.dtype == config.floatX
        theano_rng = make_theano_rng(getattr(self, 'seed', None), default_seed,
                                     which_method="binomial")

        if X.ndim == 2 and self.sync_channels:
            raise NotImplementedError()

        p = self.drop_prob

        if not hasattr(self, 'drop_prob_y') or self.drop_prob_y is None:
            yp = p
        else:
            yp = self.drop_prob_y

        batch_size = X_space.batch_size(X)

        if self.balance:
            flip = theano_rng.binomial(
                    size = (batch_size,),
                    p = 0.5,
                    n = 1,
                    dtype = X.dtype)

            yp = flip * (1-p) + (1-flip) * p

            dimshuffle_args = ['x'] * X.ndim

            if X.ndim == 2:
                dimshuffle_args[0] = 0
                assert not self.sync_channels
            else:
                dimshuffle_args[X_space.axes.index('b')] = 0
                if self.sync_channels:
                    del dimshuffle_args[X_space.axes.index('c')]

            flip = flip.dimshuffle(*dimshuffle_args)

            p = flip * (1-p) + (1-flip) * p

        # size needs to have a fixed length at compile time or the
        # theano random number generator will be angry
        size = tuple([ X.shape[i] for i in xrange(X.ndim) ])
        if self.sync_channels:
            del size[X_space.axes.index('c')]

        drop_mask = theano_rng.binomial(
                    size = size,
                    p = p,
                    n = 1,
                    dtype = X.dtype)

        X_name = make_name(X, 'anon_X')
        drop_mask.name = 'drop_mask(%s)' % X_name

        if Y is not None:
            assert isinstance(yp, float) or yp.ndim < 2
            drop_mask_Y = theano_rng.binomial(
                    size = (batch_size, ),
                    p = yp,
                    n = 1,
                    dtype = X.dtype)
            assert drop_mask_Y.ndim == 1
            Y_name = make_name(Y, 'anon_Y')
            drop_mask_Y.name = 'drop_mask_Y(%s)' % Y_name
            return drop_mask, drop_mask_Y

        return drop_mask
