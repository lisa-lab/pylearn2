"""
This module contains cost functions to use with deep Boltzmann machines
(pylearn2.models.dbm).
"""

__authors__ = ["Ian Goodfellow", "Vincent Dumoulin", "Devon Hjelm"]
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import numpy as np
import logging
import warnings

from theano.compat.python2x import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
RandomStreams = MRG_RandomStreams
from theano import tensor as T

import pylearn2
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import (
    FixedVarDescr, DefaultDataSpecsMixin, NullDataSpecsMixin
)

from pylearn2.sandbox.dbm_v2 import dbm
from pylearn2.sandbox.dbm_v2.layer import BinaryVectorMaxPool
from pylearn2.sandbox.dbm_v2 import flatten
from pylearn2.sandbox.dbm_v2.layer import BinaryVector
from pylearn2.sandbox.dbm_v2.layer import Softmax

from pylearn2 import utils
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng


logger = logging.getLogger(__name__)


# Positive phase methods

def positive_phase(model, X, Y, num_gibbs_steps=1, supervised=False,
                   theano_rng=None, method="VARIATIONAL"):
    """
    Wrapper function for positive phase.
    Method is controled by switch string "method".

    Parameters
    ----------
    X: input observables
    Y: supervised observables
    num_gibbs_steps: number of gibbs steps for sampling method
    theano_rng for sampling method
    method: method for positive phase: VARIATIONAL or SAMPLING.
    """

    if method == "VARIATIONAL":
        return variational_positive_phase(model, X, Y,
                                          supervised=supervised)
    elif method == "SAMPLING":
        return sampling_positive_phase(model, X, Y,
                                       supervised=supervised,
                                       num_gibbs_steps=num_gibbs_steps,
                                       theano_rng=theano_rng)
    else: raise ValueError("Available methods for positive phase are VARIATIONAL and SAMPLING")

def variational_positive_phase(model, X, Y, supervised):
    """
    .. todo::

    WRITEME
    """
    if supervised:
        assert Y is not None
        # note: if the Y layer changes to something without linear energy,
        # we'll need to make the expected energy clamp Y in the positive
        # phase
        assert isinstance(model.hidden_layers[-1], Softmax)

    q = model.mf(X, Y)

    """
    Use the non-negativity of the KL divergence to construct a lower
    bound on the log likelihood. We can drop all terms that are
    constant with respect to the model parameters:

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
                                disconnected_inputs='ignore')))
    return gradients

def sampling_positive_phase(model, X, Y, supervised, num_gibbs_steps, theano_rng):
    """
    .. todo::

    WRITEME
    """
    assert num_gibbs_steps is not None
    assert theano_rng is not None
    # If there's only one hidden layer, there's no point in sampling.
    if len(model.hidden_layers) == 1: num_gibbs_steps = 1
    layer_to_clamp = OrderedDict([(model.visible_layer, True)])
    layer_to_pos_samples = OrderedDict([(model.visible_layer, X)])
    if supervised:
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
        num_steps=num_gibbs_steps,
        theano_rng=theano_rng)
    q = [layer_to_pos_samples[layer] for layer in model.hidden_layers]

    pos_samples = flatten(q)

    # The gradients of the expected energy under q are easy, we can just
    # do that in theano
    expected_energy_q = model.energy(X, q).mean()
    params = list(model.get_params())
    gradients = OrderedDict(
        safe_zip(params, T.grad(expected_energy_q, params,
                                consider_constant=pos_samples,
                                disconnected_inputs='ignore')))
    return gradients

# Negative phase methods

def negative_phase(model, layer_to_chains, method="STANDARD"):
    """
    Wrapper function for negative phase.

    Parameters
    ----------
    model: a dbm model.
    layer_to_chains: dicitonary of layer chains for sampling.
    method: standard or toronto
    """

    if method == "STANDARD":
        return standard_negative_phase(model, layer_to_chains)
    elif method == "TORONTO":
        return toronto_negative_phase(model, layer_to_chains)
    else: raise ValueError("Available methods for negative phase are STANDARD and TORONTO")

def standard_negative_phase(model, layer_to_chains):
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
        [layer_to_chains[layer] for layer in model.hidden_layers]).mean()

    samples = flatten(layer_to_chains.values())
    for i, sample in enumerate(samples):
        if sample.name is None:
            sample.name = 'sample_'+str(i)

    neg_phase_grads = OrderedDict(
        safe_zip(params, T.grad(-expected_energy_p, params,
                                 consider_constant=samples,
                                 disconnected_inputs='ignore')))
    return neg_phase_grads

def toronto_negative_phase(model, layer_to_chains):
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
        V_samples, [H1_mf, H2_mf, Y_samples]).mean()

    constants = flatten([V_samples, H1_mf, H2_mf, Y_samples])

    neg_phase_grads = OrderedDict(
        safe_zip(params, T.grad(-expected_energy_p, params,
                                 consider_constant=constants)))
    return neg_phase_grads


class BaseCD(DefaultDataSpecsMixin, Cost):
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

    def __init__(self, num_chains=1, num_gibbs_steps=1, supervised=False,
                 toronto_neg=False, theano_rng=None,
                 positive_method = "SAMPLING", negative_method = "STANDARD"):
        self.__dict__.update(locals())
        del self.self

        self.theano_rng = make_theano_rng(theano_rng, 2012+10+14, which_method="binomial")
        assert supervised in [True, False]
        if toronto_neg:
            self.negative_method = "TORONTO"

    def expr(self, model, data):
        """
        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)

        return None

    def _get_positive_phase(self, model, X, Y=None):
        """
        Get positive phase.
        """
        return positive_phase(model, X, Y, supervised=self.supervised,
                              method=self.positive_method,
                              num_gibbs_steps=self.num_gibbs_steps,
                              theano_rng=self.theano_rng), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        .. todo::

            WRITEME

        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """
        layer_to_chains = model.initialize_chains(X, Y, self.theano_rng)
        updates, layer_to_chains = model.get_sampling_updates(layer_to_chains,
                                                              self.theano_rng,
                                                              num_steps=self.num_gibbs_steps,
                                                              return_layer_to_updated=True)

        neg_phase_grads = negative_phase(model, layer_to_chains, method=self.negative_method)

        return neg_phase_grads, updates

    def get_gradients(self, model, data, persistent=False):
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
        if persistent:
            for key, val in pos_updates.items():
                updates[key] = val
            for key, val in neg_updates.items():
                updates[key] = val

        gradients = OrderedDict()
        for param in list(pos_phase_grads.keys()):
            gradients[param] = neg_phase_grads[param] + pos_phase_grads[param]
        return gradients, updates

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


class VariationalCD(BaseCD):
    """
    An intractable cost representing the negative log likelihood of a DBM.
    The gradient of this bound is computed using a markov chain initialized
    with the training example.

    Source: Hinton, G. Training Products of Experts by Minimizing
            Contrastive Divergence
    """

    def __init__(self, num_gibbs_steps=2, supervised=False,
                 toronto_neg=False, theano_rng=None):
        super(VariationalCD, self).__init__(num_gibbs_steps,
                                            supervised=supervised,
                                            toronto_neg=toronto_neg,
                                            positive_method="VARIATIONAL",
                                            negative_method="STANDARD")



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
            total_cost = reduce(lambda x, y: x + y, layer_costs)
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


class L2WeightDecay(NullDataSpecsMixin, Cost):
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
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'DBM_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost


class L1WeightDecay(NullDataSpecsMixin, Cost):
    """
    A Cost that applies the following cost function:

    coeff * sum(abs(weights))
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
        layer_costs = [ layer.get_l1_weight_decay(coeff)
            for layer, coeff in safe_izip(model.hidden_layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_l1_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'DBM_L1WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'l1_weight_decay'

        return total_cost