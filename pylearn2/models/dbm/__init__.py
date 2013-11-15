"""
This module contains functionality related to deep Boltzmann machines.
They are implemented generically in order to make it easy to support
convolution versions, etc.

This code was moved piece by piece incrementally over time from Ian's
private research repository, and it is altogether possible that he
broke something or left out a piece while moving it. If you find any
problems please don't hesitate to contact pylearn-dev and we will fix
the problem and add a unit test.
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np
import sys
import time
import warnings

import theano
from theano.compat.python2x import OrderedDict
from theano import config
from theano import function
from theano import gof
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.expr.probabilistic_max_pooling import max_pool
from pylearn2.expr.probabilistic_max_pooling import max_pool_b01c
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear.conv2d import make_random_conv2D
from pylearn2.linear.conv2d import make_sparse_random_conv2D
from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.base import Block
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import block_gradient
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX

warnings.warn("DBM changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when unrolling inference don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)


class DBM(Model):
    """

    A deep Boltzmann machine.

    See "Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton
    for details.

    """

    def __init__(self,
            batch_size,
            visible_layer,
            hidden_layers,
            niter, inference_procedure=None):
        """
            batch_size:
                The batch size the model should use.
                Some convolutional LinearTransforms require a compile-time
                hardcoded batch size, otherwise this would not be part of the
                model specification.
            visible_layer:
                The visible layer of the DBM.
        """
        self.__dict__.update(locals())
        del self.self
        assert len(hidden_layers) >= 1
        self.setup_rng()
        self.layer_names = set()
        for layer in hidden_layers:
            assert layer.get_dbm() is None
            layer.set_dbm(self)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)
        self._update_layer_input_spaces()
        self.force_batch_size = batch_size
        self.freeze_set = set([])
        if inference_procedure is None:
            self.setup_inference_procedure()
        self.inference_procedure.set_dbm(self)

    def get_all_layers(self):
        return [self.visible_layer] + self.hidden_layers

    def energy(self, V, hidden):
        """
            V: a theano batch of visible unit observations
                (must be SAMPLES, not mean field parameters)
            hidden: a list, one element per hidden layer, of
                batches of samples
                (must be SAMPLES, not mean field parameters)

            returns: a vector containing the energy of each
                    sample.

            Applying this function to non-sample theano variables
            is not guaranteed to give you an expected energy
            in general, so don't use this that way.
        """

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state = V, average=False))

        assert len(self.hidden_layers) > 0 # this could be relaxed, but current code assumes it

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below = self.visible_layer.upward_state(V),
            state = hidden[0], average_below=False, average=False))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            samples_below = hidden[i-1]
            layer_below = self.hidden_layers[i-1]
            samples_below = layer_below.upward_state(samples_below)
            samples = hidden[i]
            terms.append(layer.expected_energy_term(state_below=samples_below, state=samples,
                average_below=False, average=False))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval

    def mf(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

    def expected_energy(self, V, mf_hidden):
        """
            V: a theano batch of visible unit observations
                (must be SAMPLES, not mean field parameters:
                    the random variables in the expectation are
                    the hiddens only)

            mf_hidden: a list, one element per hidden layer, of
                      batches of variational parameters
                (must be VARIATIONAL PARAMETERS, not samples.
                Layers with analytically determined variance parameters
                for their mean field parameters will use those to integrate
                over the variational distribution, so it's not generally
                the same thing as measuring the energy at a point.)

            returns: a vector containing the expected energy of
                    each example under the corresponding variational
                    distribution.
        """

        self.visible_layer.space.validate(V)
        assert isinstance(mf_hidden, (list, tuple))
        assert len(mf_hidden) == len(self.hidden_layers)

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state = V, average=False))

        assert len(self.hidden_layers) > 0 # this could be relaxed, but current code assumes it

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below=self.visible_layer.upward_state(V), average_below=False,
            state=mf_hidden[0], average=True))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer_below = self.hidden_layers[i-1]
            mf_below = mf_hidden[i-1]
            mf_below = layer_below.upward_state(mf_below)
            mf = mf_hidden[i]
            terms.append(layer.expected_energy_term(state_below=mf_below, state=mf,
                average_below=True, average=True))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval

    def setup_rng(self):
        self.rng = np.random.RandomState([2012, 10, 17])

    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = WeightDoubling()
            self.inference_procedure.set_dbm(self)

    def get_output_space(self):
        return self.hidden_layers[-1].get_output_space()

    def _update_layer_input_spaces(self):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        visible_layer = self.visible_layer
        hidden_layers = self.hidden_layers
        self.hidden_layers[0].set_input_space(visible_layer.space)
        for i in xrange(1,len(hidden_layers)):
            hidden_layers[i].set_input_space(hidden_layers[i-1].get_output_space())

    def add_layers(self, layers):
        """
            Add new layers on top of the existing hidden layers
        """

        # Patch old pickle files
        if not hasattr(self, 'rng'):
            self.setup_rng()

        hidden_layers = self.hidden_layers
        assert len(hidden_layers) > 0
        for layer in layers:
            assert layer.get_dbm() is None
            layer.set_dbm(self)
            layer.set_input_space(hidden_layers[-1].get_output_space())
            hidden_layers.append(layer)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)

    def freeze(self, parameter_set):
        # patch old pickle files
        if not hasattr(self, 'freeze_set'):
            self.freeze_set = set([])

        self.freeze_set = self.freeze_set.union(parameter_set)

    def get_params(self):

        rval = []
        for param in self.visible_layer.get_params():
            assert param.name is not None
        rval = self.visible_layer.get_params()
        for layer in self.hidden_layers:
            for param in layer.get_params():
                if param.name is None:
                    raise ValueError("All of your parameters should have names, but one of "+layer.layer_name+"'s doesn't")
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        # Patch pickle files that predate the freeze_set feature
        if not hasattr(self, 'freeze_set'):
            self.freeze_set = set([])

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.hidden_layers:
            layer.set_batch_size(batch_size)

        if not hasattr(self, 'inference_procedure'):
            self.setup_inference_procedure()
        self.inference_procedure.set_batch_size(batch_size)

    def censor_updates(self, updates):
        self.visible_layer.censor_updates(updates)
        for layer in self.hidden_layers:
            layer.censor_updates(updates)

    def get_input_space(self):
        return self.visible_layer.space

    def get_lr_scalers(self):
        rval = OrderedDict()

        params = self.get_params()

        for layer in self.hidden_layers + [ self.visible_layer ]:
            contrib = layer.get_lr_scalers()

            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)
        assert all([isinstance(val, float) for val in rval.values()])

        return rval

    def get_weights(self):
        return self.hidden_layers[0].get_weights()

    def get_weights_view_shape(self):
        return self.hidden_layers[0].get_weights_view_shape()

    def get_weights_format(self):
        return self.hidden_layers[0].get_weights_format()

    def get_weights_topo(self):
        return self.hidden_layers[0].get_weights_topo()

    def make_layer_to_state(self, num_examples, rng=None):

        """ Makes and returns a dictionary mapping layers to states.
            By states, we mean here a real assignment, not a mean field state.
            For example, for a layer containing binary random variables, the
            state will be a shared variable containing values in {0,1}, not
            [0,1].
            The visible layer will be included.
            Uses a dictionary so it is easy to unambiguously index a layer
            without needing to remember rules like vis layer = 0, hiddens start
            at 1, etc.
        """

        # Make a list of all layers
        layers = [self.visible_layer] + self.hidden_layers

        if rng is None:
            rng = self.rng

        states = [layer.make_state(num_examples, rng) for layer in layers]

        zipped = safe_zip(layers, states)

        def recurse_check(layer, state):
            if isinstance(state, (list, tuple)):
                for elem in state:
                    recurse_check(layer, elem)
            else:
                val = state.get_value()
                m = val.shape[0]
                if m != num_examples:
                    raise ValueError(layer.layer_name+" gave state with "+str(m)+ \
                            " examples in some component. We requested "+str(num_examples))

        for layer, state in zipped:
            recurse_check(layer, state)

        rval = OrderedDict(zipped)

        return rval

    def make_layer_to_symbolic_state(self, num_examples, rng=None):

        """
        Makes and returns a dictionary mapping layers to states.
        By states, we mean here a real assignment, not a mean field state.
        For example, for a layer containing binary random variables, the
        state will be a symbolic variable containing values in {0,1}, not
        [0,1].
        The visible layer will be included.
        Uses a dictionary so it is easy to unambiguously index a layer
        without needing to remember rules like vis layer = 0, hiddens start
        at 1, etc.
        """

        # Make a list of all layers
        layers = [self.visible_layer] + self.hidden_layers

        assert rng is not None

        states = [layer.make_symbolic_state(num_examples, rng) for layer in layers]

        zipped = safe_zip(layers, states)

        rval = OrderedDict(zipped)

        return rval

    def mcmc_steps(self, layer_to_state, theano_rng, layer_to_clamp = None,
            num_steps = 1):
        """
            layer_to_state: a dictionary mapping the SuperDBM_Layer instances
                            contained in self to theano variables representing
                            batches of samples of them.
            theano_rng: a MRG_RandomStreams object
            layer_to_clamp: (optional) a dictionary mapping layers to bools
                            if a layer is not in the dictionary, defaults to False
                            True indicates that this layer should be clamped, so
                            we are sampling from a conditional distribution rather
                            than the joint
            returns:
                layer_to_updated_state
                    dict mapping layers to theano variables representing the updated
                    samples

            The specific sampling schedule used is to sample all of the even-idexed
            layers of model.hidden_layers, then the visible layer and all the odd-indexed
            layers.
        """

        # Validate num_steps
        assert isinstance(num_steps, py_integer_types)
        assert num_steps > 0

        # Implement the num_steps > 1 case by repeatedly calling the num_steps == 1 case
        if num_steps != 1:
            for i in xrange(num_steps):
                layer_to_state = self.mcmc_steps(layer_to_state, theano_rng, layer_to_clamp,
                        num_steps = 1)
            return layer_to_state

        # The rest of the function is the num_steps = 1 case

        assert len(self.hidden_layers) > 0 # current code assumes this, though we could certainly
                                           # relax this constraint

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert key is self.visible_layer or key in self.hidden_layers

        for layer in [self.visible_layer] + self.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        #Assemble the return value
        layer_to_updated = OrderedDict()

        for i, this_layer in list(enumerate(self.hidden_layers))[::2]:
            # Iteration i does the Gibbs step for hidden_layers[i]

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            if i == 0:
                layer_below = self.visible_layer
            else:
                layer_below = self.hidden_layers[i-1]
            state_below = layer_to_state[layer_below]
            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.hidden_layers):
                layer_above = self.hidden_layers[i + 1]
                state_above = layer_to_state[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                state_above = None
                layer_above = None

            if layer_to_clamp[this_layer]:
                this_state = layer_to_state[this_layer]
                this_sample = this_state
            else:
                # Compute the Gibbs sampling update
                # Sample the state of this layer conditioned
                # on its Markov blanket (the layer above and
                # layer below)
                this_sample = this_layer.sample(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above,
                        theano_rng = theano_rng)

            layer_to_updated[this_layer] = this_sample

        #Sample the visible layer
        vis_state = layer_to_state[self.visible_layer]
        if layer_to_clamp[self.visible_layer]:
            vis_sample = vis_state
        else:
            first_hid = self.hidden_layers[0]
            state_above = layer_to_updated[first_hid]
            state_above = first_hid.downward_state(state_above)

            vis_sample = self.visible_layer.sample(
                    state_above = state_above,
                    layer_above = first_hid,
                    theano_rng = theano_rng)
        layer_to_updated[self.visible_layer] = vis_sample

        # Sample the odd-numbered layers
        for i, this_layer in list(enumerate(self.hidden_layers))[1::2]:

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            layer_below = self.hidden_layers[i-1]

            # We want to sample from each conditional distribution
            # ***sequentially*** so we must use the updated version
            # of the state for the layers whose updates we have
            # calculcated already, in layer_to_updated.
            # If we used the original value from
            # layer_to_state
            # then we would sample from each conditional
            # ***simultaneously*** which does not implement MCMC
            # sampling.
            state_below = layer_to_updated[layer_below]

            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.hidden_layers):
                layer_above = self.hidden_layers[i + 1]
                state_above = layer_to_updated[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                state_above = None
                layer_above = None

            if layer_to_clamp[this_layer]:
                this_state = layer_to_state[this_layer]
                this_sample = this_state
            else:
                # Compute the Gibbs sampling update
                # Sample the state of this layer conditioned
                # on its Markov blanket (the layer above and
                # layer below)
                this_sample = this_layer.sample(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above,
                        theano_rng = theano_rng)

            layer_to_updated[this_layer] = this_sample

        # Check that all layers were updated
        assert all([layer in layer_to_updated for layer in layer_to_state])
        # Check that we didn't accidentally treat any other object as a layer
        assert all([layer in layer_to_state for layer in layer_to_updated])
        # Check that clamping worked
        assert all([(layer_to_state[layer] is layer_to_updated[layer]) == \
                layer_to_clamp[layer] for layer in layer_to_state])

        return layer_to_updated

    def get_sampling_updates(self, layer_to_state, theano_rng,
            layer_to_clamp = None, num_steps = 1, return_layer_to_updated = False):
        """
            This method is for getting an updates dictionary for a theano function.
            It thus implies that the samples are represented as shared variables.
            If you want an expression for a sampling step applied to arbitrary
            theano variables, use the 'mcmc_steps' method. This is a wrapper around
            that method.

            Parameters
            ----------
            layer_to_state: a dictionary mapping the SuperDBM_Layer instances
                            contained in self to shared variables representing
                            batches of samples of them.
                            (you can allocate one by calling
                            self.make_layer_to_state)
            theano_rng: a MRG_RandomStreams object
            layer_to_clamp: (optional) a dictionary mapping layers to bools
                            if a layer is not in the dictionary, defaults to False
                            True indicates that this layer should be clamped, so
                            we are sampling from a conditional distribution rather
                            than the joint
            returns a dictionary mapping each shared variable to an expression
                     to update it. Repeatedly applying these updates does MCMC
                     sampling.

            The specific sampling schedule used is to sample all of the even-idexed
            layers of model.hidden_layers, then the visible layer and all the odd-indexed
            layers.
        """

        updated = self.mcmc_steps(layer_to_state, theano_rng, layer_to_clamp, num_steps)

        rval = OrderedDict()

        def add_updates(old, new):
            if isinstance(old, (list, tuple)):
                for old_elem, new_elem in safe_izip(old, new):
                    add_updates(old_elem, new_elem)
            else:
                rval[old] = new

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert key is self.visible_layer or key in self.hidden_layers

        for layer in [self.visible_layer] + self.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        # Translate update expressions into theano updates
        for layer in layer_to_state:
            old = layer_to_state[layer]
            new = updated[layer]
            if layer_to_clamp[layer]:
                assert new is old
            else:
                add_updates(old, new)

        assert isinstance(self.hidden_layers, list)

        if return_layer_to_updated:
            return rval, updated

        return rval

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)
        X = data
        history = self.mf(X, return_history = True)
        q = history[-1]

        rval = OrderedDict()

        ch = self.visible_layer.get_monitoring_channels()
        for key in ch:
            rval['vis_'+key] = ch[key]

        for state, layer in safe_zip(q, self.hidden_layers):
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            ch = layer.get_monitoring_channels_from_state(state)
            for key in ch:
                rval['mf_'+layer.layer_name+'_'+key]  = ch[key]

        if len(history) > 1:
            prev_q = history[-2]

            flat_q = flatten(q)
            flat_prev_q = flatten(prev_q)

            mx = None
            for new, old in safe_zip(flat_q, flat_prev_q):
                cur_mx = abs(new - old).max()
                if new is old:
                    print new, 'is', old
                    assert False
                if mx is None:
                    mx = cur_mx
                else:
                    mx = T.maximum(mx, cur_mx)

            rval['max_var_param_diff'] = mx

            for layer, new, old in safe_zip(self.hidden_layers,
                q, prev_q):
                sum_diff = 0.
                for sub_new, sub_old in safe_zip(flatten(new), flatten(old)):
                    sum_diff += abs(sub_new - sub_old).sum()
                denom = self.batch_size * layer.get_total_state_space().get_total_dimension()
                denom = np.cast[config.floatX](denom)
                rval['mean_'+layer.layer_name+'_var_param_diff'] = sum_diff / denom

        return rval

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channel.

        This implementation returns specification corresponding to unlabeled
        inputs.
        """
        return (self.get_input_space(), self.get_input_source())

    def get_test_batch_size(self):
        return self.batch_size

    def reconstruct(self, V):

        H = self.mf(V)[0]

        downward_state = self.hidden_layers[0].downward_state(H)

        recons = self.visible_layer.inpaint_update(
                layer_above = self.hidden_layers[0],
                state_above = downward_state,
                drop_mask = None, V = None)

        return recons


class Layer(Model):
    """
    Abstract class.
    A layer of a DBM.
    May only belong to one DBM.

    Each layer has a state ("total state") that can be split into
    the piece that is visible to the layer above ("upward state")
    and the piece that is visible to the layer below ("downward state").
    (Since visible layers don't have a downward state, the downward_state
    method only appears in the DBM_HiddenLayer subclass)

    For simple layers, all three of these are the same thing.
    """

    def get_dbm(self):
        """
        Returns the DBM that this layer belongs to, or None
        if it has not been assigned to a DBM yet.
        """

        if hasattr(self, 'dbm'):
            return self.dbm

        return None

    def set_dbm(self, dbm):
        """
        Assigns this layer to a DBM.
        """
        assert self.get_dbm() is None
        self.dbm = dbm

    def get_total_state_space(self):
        """
        Returns the Space that the layer's total state lives in.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +\
                "get_total_state_space()")


    def get_monitoring_channels(self):
        """
        TODO WRITME
        """
        return OrderedDict()

    def get_monitoring_channels_from_state(self, state):
        """
        TODO WRITEME
        """
        return OrderedDict()

    def upward_state(self, total_state):
        """
            Takes total_state and turns it into the state that layer_above should
            see when computing P( layer_above | this_layer).

            So far this has two uses:
                If this layer consists of a detector sub-layer h that is pooled
                into a pooling layer p, then total_state = (p,h) but
                layer_above should only see p.

                If the conditional P( layer_above | this_layer) depends on
                parameters of this_layer, sometimes you can play games with
                the state to avoid needing the layers to communicate. So far
                the only instance of this usage is when the visible layer
                is N( Wh, beta). This makes the hidden layer be
                sigmoid( v beta W + b). Rather than having the hidden layer
                explicitly know about beta, we can just pass v beta as
                the upward state.

            Note: this method should work both for computing sampling updates
            and for computing mean field updates. So far I haven't encountered
            a case where it needs to do different things for those two
            contexts.
        """
        return total_state

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        raise NotImplementedError("%s doesn't implement make_state" %
                type(self))

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        Returns a theano symbolic variable containing an actual state (not a
        mean field state) for this variable.
        """

        raise NotImplementedError("%s doesn't implement make_symbolic_state" %
                                  type(self))

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
            state_below is layer_below.upward_state(full_state_below)
            where full_state_below is the same kind of object as you get
            out of layer_below.make_state

            state_above is layer_above.downward_state(full_state_above)

            theano_rng is an MRG_RandomStreams instance

            Returns an expression for samples of this layer's state,
            conditioned on the layers above and below
            Should be valid as an update to the shared variable returned
            by self.make_state

            Note: this can return multiple expressions if this layer's
            total state consists of more than one shared variable
        """

        if hasattr(self, 'get_sampling_updates'):
            raise AssertionError("Looks like "+str(type(self))+" needs to rename get_sampling_updates to sample.")

        raise NotImplementedError("%s doesn't implement sample" %
                type(self))

    def expected_energy_term(self, state,
                                   average,
                                   state_below,
                                   average_below):
        """

            Returns a term of the expected energy of the entire model.
            This term should correspond to the expected value of terms
            of the energy function that:
                -involve this layer only
                -if there is a layer below, include terms that
                 involve both this layer and the layer below

            Do not include terms that involve the layer below only.
            Do not include any terms that involve the layer above, if it
            exists, in any way (the interface doesn't let you see the layer
            above anyway).

            Parameters
            ----------
            state_below: the upward state of the layer below.
            state: the total state of this layer

            average_below: if True, the layer below is one of the variables to
                integrate over in the expectation, and state_below gives its
                variational parameters. if False, that layer is to be held constant
                and state_below gives a set of assignments to it
            average: like average_below, but for 'state' rather than 'state_below'

            returns: a 1-d theano tensor giving the expected energy term for each example


        """
        raise NotImplementedError(str(type(self))+" does not implement expected_energy_term.")


class VisibleLayer(Layer):
    """
    Abstract class.
    A layer of a DBM that may be used as a visible layer.
    Currently, all implemented layer classes may be either visible
    or hidden but not both. It may be worth making classes that can
    play both roles though. This would allow getting rid of the BinaryVector
    class.
    """

    def get_total_state_space(self):
        return self.get_input_space()

class HiddenLayer(Layer):
    """
    Abstract class.
    A layer of a DBM that may be used as a hidden layer.
    """

    def downward_state(self, total_state):
        return total_state

    def get_stdev_rewards(self, state, coeffs):
        raise NotImplementedError(str(type(self))+" does not implement get_stdev_rewards")

    def get_range_rewards(self, state, coeffs):
        raise NotImplementedError(str(type(self))+" does not implement get_range_rewards")

    def get_l1_act_cost(self, state, target, coeff, eps):
        raise NotImplementedError(str(type(self))+" does not implement get_l1_act_cost")

    def get_l2_act_cost(self, state, target, coeff):
        raise NotImplementedError(str(type(self))+" does not implement get_l2_act_cost")

def init_sigmoid_bias_from_marginals(dataset, use_y = False):
    """
    Returns b such that sigmoid(b) has the same marginals as the
    data. Assumes dataset contains a design matrix. If use_y is
    true, sigmoid(b) will have the same marginals as the targets,
    rather than the features.
    """
    if use_y:
        X = dataset.y
    else:
        X = dataset.get_design_matrix()
    return init_sigmoid_bias_from_array(X)

def init_sigmoid_bias_from_array(arr):
    X = arr
    if not (X.max() == 1):
        raise ValueError("Expected design matrix to consist entirely "
                "of 0s and 1s, but maximum value is "+str(X.max()))
    if X.min() != 0.:
        raise ValueError("Expected design matrix to consist entirely of "
                "0s and 1s, but minimum value is "+str(X.min()))
    # removed this check so we can initialize the marginals
    # with a dataset of bernoulli params
    # assert not np.any( (X > 0.) * (X < 1.) )

    mean = X.mean(axis=0)

    mean = np.clip(mean, 1e-7, 1-1e-7)

    init_bias = inverse_sigmoid_numpy(mean)

    return init_bias


class BinaryVector(VisibleLayer):
    """
    A DBM visible layer consisting of binary random variables living
    in a VectorSpace.
    """

    def __init__(self,
            nvis,
            bias_from_marginals = None,
            center = False,
            copies = 1):
        """
            nvis: the dimension of the space
            bias_from_marginals: a dataset, whose marginals are used to
                            initialize the visible biases
        """

        self.__dict__.update(locals())
        del self.self
        # Don't serialize the dataset
        del self.bias_from_marginals

        self.space = VectorSpace(nvis)
        self.input_space = self.space

        origin = self.space.get_origin()

        if bias_from_marginals is None:
            init_bias = np.zeros((nvis,))
        else:
            init_bias = init_sigmoid_bias_from_marginals(bias_from_marginals)

        self.bias = sharedX(init_bias, 'visible_bias')

        if center:
            self.offset = sharedX(sigmoid_numpy(init_bias))

    def get_biases(self):
        return self.bias.get_value()

    def set_biases(self, biases, recenter=False):
        self.bias.set_value(biases)
        if recenter:
            assert self.center
            self.offset.set_value(sigmoid_numpy(self.bias.get_value()))

    def upward_state(self, total_state):

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            rval = total_state - self.offset
        else:
            rval = total_state

        if not hasattr(self, 'copies'):
            self.copies = 1

        return rval * self.copies


    def get_params(self):
        return [self.bias]

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):


        assert state_below is None
        if self.copies != 1:
            raise NotImplementedError()

        msg = layer_above.downward_message(state_above)

        bias = self.bias

        z = msg + bias

        phi = T.nnet.sigmoid(z)

        rval = theano_rng.binomial(size = phi.shape, p = phi, dtype = phi.dtype,
                       n = 1 )

        return rval

    def mf_update(self, state_above, layer_above):
        msg = layer_above.downward_message(state_above)
        mu = self.bias

        z = msg + mu

        rval = T.nnet.sigmoid(z)

        return rval


    def make_state(self, num_examples, numpy_rng):
        if not hasattr(self, 'copies'):
            self.copies = 1
        if self.copies != 1:
            raise NotImplementedError()
        driver = numpy_rng.uniform(0.,1., (num_examples, self.nvis))
        mean = sigmoid_numpy(self.bias.get_value())
        sample = driver < mean

        rval = sharedX(sample, name = 'v_sample_shared')

        return rval

    def make_symbolic_state(self, num_examples, theano_rng):
        if not hasattr(self, 'copies'):
            self.copies = 1
        if self.copies != 1:
            raise NotImplementedError()
        mean = T.nnet.sigmoid(self.bias)
        rval = theano_rng.binomial(size=(num_examples, self.nvis), p=mean)

        return rval

    def expected_energy_term(self, state, average, state_below = None, average_below = None):

        if self.center:
            state = state - self.offset

        assert state_below is None
        assert average_below is None
        assert average in [True, False]
        self.space.validate(state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        rval = -T.dot(state, self.bias)

        assert rval.ndim == 1

        return rval * self.copies

class BinaryVectorMaxPool(HiddenLayer):
    """
        A hidden layer that does max-pooling on binary vectors.
        It has two sublayers, the detector layer and the pooling
        layer. The detector layer is its downward state and the pooling
        layer is its upward state.

        TODO: this layer uses (pooled, detector) as its total state,
              which can be confusing when listing all the states in
              the network left to right. Change this and
              pylearn2.expr.probabilistic_max_pooling to use
              (detector, pooled)
    """

    def __init__(self,
             detector_layer_dim,
            pool_size,
            layer_name,
            irange = None,
            sparse_init = None,
            sparse_stdev = 1.,
            include_prob = 1.0,
            init_bias = 0.,
            W_lr_scale = None,
            b_lr_scale = None,
            center = False,
            mask_weights = None,
            max_col_norm = None,
            copies = 1):
        """

            include_prob: probability of including a weight element in the set
                    of weights initialized to U(-irange, irange). If not included
                    it is initialized to 0.

        """
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if self.center:
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset = sharedX(sigmoid_numpy(self.b.get_value()))

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not (self.detector_layer_dim % self.pool_size == 0):
            raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                    (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = self.detector_layer_dim / self.pool_size
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.dbm.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                                 self.irange,
                                 (self.input_dim, self.detector_layer_dim)) * \
                    (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                     < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None
        if not hasattr(self, 'max_col_norm'):
            self.max_col_norm = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))


    def get_total_state_space(self):
        return CompositeSpace((self.output_space, self.h_space))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        return W.get_value()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter = False):
        self.b.set_value(biases)
        if recenter:
            assert self.center
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset.set_value(sigmoid_numpy(self.b.get_value()))

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decidew how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total / cols
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W ,= self.transformer.get_params()

        W = W.T

        W = W.reshape((self.detector_layer_dim, self.input_space.shape[0],
            self.input_space.shape[1], self.input_space.nchannels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def upward_state(self, total_state):
        p,h = total_state
        self.h_space.validate(h)
        self.output_space.validate(p)

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            return p - self.offset

        if not hasattr(self, 'copies'):
            self.copies = 1

        return p * self.copies

    def downward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            return h - self.offset

        return h * self.copies

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
              ('row_norms_min'  , row_norms.min()),
              ('row_norms_mean' , row_norms.mean()),
              ('row_norms_max'  , row_norms.max()),
              ('col_norms_min'  , col_norms.min()),
              ('col_norms_mean' , col_norms.mean()),
              ('col_norms_max'  , col_norms.max()),
            ])


    def get_monitoring_channels_from_state(self, state):

        P, H = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_'), (H, 'h_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                    ('max_x.max_u', v_max.max()),
                    ('max_x.mean_u', v_max.mean()),
                    ('max_x.min_u', v_max.min()),
                    ('min_x.max_u', v_min.max()),
                    ('min_x.mean_u', v_min.mean()),
                    ('min_x.min_u', v_min.min()),
                    ('range_x.max_u', v_range.max()),
                    ('range_x.mean_u', v_range.mean()),
                    ('range_x.min_u', v_range.min()),
                    ('mean_x.max_u', v_mean.max()),
                    ('mean_x.mean_u', v_mean.mean()),
                    ('mean_x.min_u', v_mean.min())
                    ]:
                rval[prefix+key] = val

        return rval

    def get_stdev_rewards(self, state, coeffs):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if isinstance(coeffs, str):
                coeffs = float(coeffs)
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            assert all([isinstance(elem, float) for elem in [c]])
            if c == 0.:
                continue
            mn = s.mean(axis=0)
            dev = s - mn
            stdev = T.sqrt(T.sqr(dev).mean(axis=0))
            rval += (0.5 - stdev).mean()*c

        return rval
    def get_range_rewards(self, state, coeffs):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if isinstance(coeffs, str):
                coeffs = float(coeffs)
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            assert all([isinstance(elem, float) for elem in [c]])
            if c == 0.:
                continue
            mx = s.max(axis=0)
            assert hasattr(mx.owner.op, 'grad')
            assert mx.ndim == 1
            mn = s.min(axis=0)
            assert hasattr(mn.owner.op, 'grad')
            assert mn.ndim == 1
            r = mx - mn
            rval += (1 - r).mean()*c

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps = None):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if not isinstance(target, float):
                raise TypeError("BinaryVectorMaxPool.get_l1_act_cost expected target of type float " + \
                        " but an instance named "+self.layer_name + " got target "+str(target) + " of type "+str(type(target)))
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
            if eps is None:
                eps = [0.]
            else:
                eps = [eps]
        else:
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if eps is None:
                eps = [0., 0.]
            if target[1] > target[0]:
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            assert all([isinstance(elem, float) or hasattr(elem, 'dtype') for elem in [t, c, e]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_l2_act_cost(self, state, target, coeff):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            if not isinstance(target, float):
                raise TypeError("BinaryVectorMaxPool.get_l1_act_cost expected target of type float " + \
                        " but an instance named "+self.layer_name + " got target "+str(target) + " of type "+str(type(target)))
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
        else:
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            if target[1] > target[0]:
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c in safe_zip(state, target, coeff):
            assert all([isinstance(elem, float) or hasattr(elem, 'dtype') for elem in [t, c]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.square(m-t).mean()*c

        return rval

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        if self.copies != 1:
            raise NotImplementedError()

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to None so that it may appear after layer_above / state_above in the list.")

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
        else:
            msg = None

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b
        p, h, p_sample, h_sample = max_pool_channels(z,
                self.pool_size, msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        self.h_space.validate(downward_state)
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval * self.copies

    def init_mf_state(self):
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size, self.detector_layer_dim).astype(self.b.dtype) + \
                self.b.dimshuffle('x', 0)
        rval = max_pool_channels(z = z,
                pool_size = self.pool_size)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        if not hasattr(self, 'copies'):
            self.copies = 1

        if self.copies != 1:
            raise NotImplementedError()


        empty_input = self.h_space.get_origin_batch(num_examples)
        empty_output = self.output_space.get_origin_batch(num_examples)

        h_state = sharedX(empty_input)
        p_state = sharedX(empty_output)

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        default_z = T.zeros_like(h_state) + self.b

        p_exp, h_exp, p_sample, h_sample = max_pool_channels(
                z = default_z,
                pool_size = self.pool_size,
                theano_rng = theano_rng)

        assert h_sample.dtype == default_z.dtype

        f = function([], updates = [
            (p_state , p_sample),
            (h_state , h_sample)
            ])

        f()

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        Returns a theano symbolic variable containing an actual state
        (not a mean field state) for this variable.
        """

        if not hasattr(self, 'copies'):
            self.copies = 1

        if self.copies != 1:
            raise NotImplementedError()

        default_z = T.alloc(self.b, num_examples, self.detector_layer_dim)

        p_exp, h_exp, p_sample, h_sample = max_pool_channels(z=default_z,
                                                             pool_size=self.pool_size,
                                                             theano_rng=theano_rng)

        assert h_sample.dtype == default_z.dtype

        return p_sample, h_sample

    def expected_energy_term(self, state, average, state_below, average_below):

        # Don't need to do anything special for centering, upward_state / downward state
        # make it all just work

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = T.dot(downward_state, self.b)
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=1)

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval * self.copies

    def linear_feed_forward_approximation(self, state_below):
        """
        Used to implement TorontoSparsity. Unclear exactly what properties of it are
        important or how to implement it for other layers.

        Properties it must have:
            output is same kind of data structure (ie, tuple of theano 2-tensors)
            as mf_update

        Properties it probably should have for other layer types:
            An infinitesimal change in state_below or the parameters should cause the same sign of change
            in the output of linear_feed_forward_approximation and in mf_update

            Should not have any non-linearities that cause the gradient to shrink

            Should disregard top-down feedback
        """

        z = self.transformer.lmul(state_below) + self.b

        if self.pool_size != 1:
            # Should probably implement sum pooling for the non-pooled version,
            # but in reality it's not totally clear what the right answer is
            raise NotImplementedError()

        return z, z

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = max_pool_channels(z, self.pool_size, msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h

class Softmax(HiddenLayer):

    presynaptic_name = "presynaptic_Y_hat"

    def __init__(self, n_classes, layer_name, irange = None,
                 sparse_init = None, sparse_istdev = 1., W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 copies = 1, center = False,
                 learn_init_inpainting_state = True):
        """
            copies: We regard the layer as being replicated so that there
                   are <copies> instances of it.
                   All sample and mean field states are the *average* of
                   all of these copies, and the weights to each copy are
                   tied.
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self

        assert isinstance(n_classes, py_integer_types)

        self.output_space = VectorSpace(n_classes)
        self.b = sharedX( np.zeros((n_classes,)), name = 'softmax_b')

        if self.center:
            b = self.b.get_value()
            self.offset = sharedX(np.exp(b) / np.exp(b).sum())

    def censor_updates(self, updates):

        if not hasattr(self, 'max_col_norm'):
            self.max_col_norm = None

        if self.max_col_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_lr_scalers(self):

        rval = OrderedDict()

        # Patch old pickle files
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_total_state_space(self):
        return self.output_space

    def get_monitoring_channels_from_state(self, state):

        mx = state.max(axis=1)

        return OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        self.desired_space = VectorSpace(self.input_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.dbm.rng

        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_classes))
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.n_classes))
            for i in xrange(self.n_classes):
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0.:
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn() * self.sparse_istdev

        self.W = sharedX(W,  'softmax_W' )

        self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases, recenter=False):
        self.b.set_value(biases)
        if recenter:
            assert self.center
            self.offset.set_value( (np.exp(biases) / np.exp(biases).sum()).astype(self.offset.dtype))

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):


        if self.copies != 1:
            raise NotImplementedError("need to draw self.copies samples and average them together.")

        if state_above is not None:
            # If you implement this case, also add a unit test for it.
            # Or at least add a warning that it is not tested.
            raise NotImplementedError()

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to None so that it may appear after layer_above / state_above in the list.")

        self.input_space.validate(state_below)

        # patch old pickle files
        if not hasattr(self, 'needs_reformat'):
            self.needs_reformat = self.needs_reshape
            del self.needs_reshape

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        self.desired_space.validate(state_below)


        z = T.dot(state_below, self.W) + self.b
        h_exp = T.nnet.softmax(z)
        h_sample = theano_rng.multinomial(pvals = h_exp, dtype = h_exp.dtype)

        return h_sample

    def mf_update(self, state_below, state_above = None, layer_above = None, double_weights = False, iter_name = None):
        if state_above is not None:
            raise NotImplementedError()

        if double_weights:
            raise NotImplementedError()

        self.input_space.validate(state_below)

        # patch old pickle files
        if not hasattr(self, 'needs_reformat'):
            self.needs_reformat = self.needs_reshape
            del self.needs_reshape

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if value.shape[0] != self.dbm.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)

        assert self.W.ndim == 2
        assert state_below.ndim == 2

        b = self.b

        Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            assert value.shape[0] == self.dbm.batch_size

        return rval

    def downward_message(self, downward_state):

        if not hasattr(self, 'copies'):
            self.copies = 1

        rval =  T.dot(downward_state, self.W.T) * self.copies

        rval = self.desired_space.format_as(rval, self.input_space)

        return rval

    def recons_cost(self, Y, Y_hat_unmasked, drop_mask_Y, scale):
        """
            scale is because the visible layer also goes into the
            cost. it uses the mean over units and examples, so that
            the scale of the cost doesn't change too much with batch
            size or example size.
            we need to multiply this cost by scale to make sure that
            it is put on the same scale as the reconstruction cost
            for the visible units. ie, scale should be 1/nvis
        """


        Y_hat = Y_hat_unmasked
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z ,= owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        masked = log_prob_of * drop_mask_Y
        assert masked.ndim == 1

        rval = masked.mean() * scale * self.copies

        return - rval

    def init_mf_state(self):
        rval =  T.nnet.softmax(self.b.dimshuffle('x', 0)) + T.alloc(0., self.dbm.batch_size, self.n_classes).astype(config.floatX)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        if self.copies != 1:
            raise NotImplementedError("need to make self.copies samples and average them together.")

        t1 = time.time()

        empty_input = self.output_space.get_origin_batch(num_examples)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.b

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        h_exp = T.nnet.softmax(default_z)

        h_sample = theano_rng.multinomial(pvals = h_exp, dtype = h_exp.dtype)

        h_state = sharedX( self.output_space.get_origin_batch(
            num_examples))


        t2 = time.time()

        f = function([], updates = [(
            h_state , h_sample
            )])

        t3 = time.time()

        f()

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        h_state.name = 'softmax_sample_shared'

        return h_state

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        Returns a symbolic variable containing an actual state
        (not a mean field state) for this variable.
        """

        if self.copies != 1:
            raise NotImplementedError("need to make self.copies samples and average them together.")

        default_z = T.alloc(self.b, num_examples, self.n_classes)

        h_exp = T.nnet.softmax(default_z)

        h_sample = theano_rng.multinomial(pvals=h_exp, dtype=h_exp.dtype)

        return h_sample

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def upward_state(self, state):
        if self.center:
            return state - self.offset
        return state

    def downward_state(self, state):
        if not hasattr(self, 'center'):
            self.center = False
        if self.center:
            warnings.warn("TODO: write a unit test verifying that inference or sampling "
                    "below a centered Softmax layer works")
            return state - self.offset
        return state

    def expected_energy_term(self, state, average, state_below, average_below):

        if self.center:
            state = state - self.offset

        self.input_space.validate(state_below)
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)
        self.desired_space.validate(state_below)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = T.dot(state, self.b)
        weights_term = (T.dot(state_below, self.W) * state).sum(axis=1)

        rval = -bias_term - weights_term

        rval *= self.copies

        assert rval.ndim == 1

        return rval

    def init_inpainting_state(self, Y, noise):
        if noise:
            theano_rng = MRG_RandomStreams(2012+10+30)
            return T.nnet.softmax(theano_rng.normal(avg=0., size=Y.shape, std=1., dtype='float32'))
        rval =  T.nnet.softmax(self.b)
        if not hasattr(self, 'learn_init_inpainting_state'):
            self.learn_init_inpainting_state = 1
        if not self.learn_init_inpainting_state:
            rval = block_gradient(rval)
        return rval

    def install_presynaptic_outputs(self, outputs_dict, batch_size):

        assert self.presynaptic_name not in outputs_dict
        outputs_dict[self.presynaptic_name] = self.output_space.make_shared_batch(batch_size, self.presynaptic_name)

class InferenceProcedure(object):
    """
    TODO WRITEME
    """

    def set_dbm(self, dbm):
        self.dbm = dbm

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
        """
        TODO WRITEME
        """

        raise NotImplementedError(str(type(self))+" does not implement mf.")

    def set_batch_size(self, batch_size):
        """
        If the inference procedure is dependent on a batch size at all, makes the
        necessary internal configurations to work with that batch size.
        """

class WeightDoubling(InferenceProcedure):

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            dbm.hidden_layers[-1].get_output_space().validate(Y)

        if niter is None:
            niter = dbm.niter

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V)))

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            state_above = dbm.hidden_layers[-1].downward_state(Y)
            layer_above = dbm.hidden_layers[-1]
            assert len(dbm.hidden_layers) > 1

            # Last layer before Y does not need its weights doubled
            # because it already has top down input
            if len(dbm.hidden_layers) > 2:
                state_below = dbm.hidden_layers[-3].upward_state(H_hat[-3])
            else:
                state_below = dbm.visible_layer.upward_state(V)

            H_hat[-2] = dbm.hidden_layers[-2].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

            # Last layer is clamped to Y
            H_hat[-1] = Y



        if block_grad == 1:
            H_hat = block(H_hat)

        history = [ list(H_hat) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(1, niter):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = dbm.visible_layer.upward_state(V)
                    else:
                        state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

                if Y is not None:
                    H_hat[-1] = Y

                for j in xrange(1,len(H_hat),2):
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)
                    #end ifelse
                #end for odd layer

                if Y is not None:
                    H_hat[-1] = Y

                if block_grad == i:
                    H_hat = block(H_hat)

                history.append(list(H_hat))
            # end for mf iter
        # end if recurrent

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)
        if Y is not None:
            inferred = H_hat[:-1]
        else:
            inferred = H_hat
        for elem in flatten(inferred):
            # This check doesn't work with ('c', 0, 1, 'b') because 'b' is no longer axis 0
            # for value in get_debug_values(elem):
            #    assert value.shape[0] == dbm.batch_size
            assert V in gof.graph.ancestors([elem])
            if Y is not None:
                assert Y in gof.graph.ancestors([elem])
        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat


class DBMSampler(Block):
    """
    A Block used to sample from the last layer of a DBM with one hidden layer.
    """
    def __init__(self, dbm):
        super(DBMSampler, self).__init__()
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        self.dbm = dbm
        assert len(self.dbm.hidden_layers) == 1

    def __call__(self, inputs):
        space = self.dbm.get_input_space()
        num_examples = space.batch_size(inputs)

        last_layer = self.dbm.get_all_layers()[-1]
        layer_to_chains = self.dbm.make_layer_to_symbolic_state(
            num_examples, self.theano_rng)
        # The examples are used to initialize the visible layer's chains
        layer_to_chains[self.dbm.visible_layer] = inputs

        layer_to_clamp = OrderedDict([(self.dbm.visible_layer, True)])
        layer_to_chains = self.dbm.mcmc_steps(layer_to_chains, self.theano_rng,
                                              layer_to_clamp=layer_to_clamp,
                                              num_steps=1)

        rval = layer_to_chains[last_layer]
        rval = last_layer.upward_state(rval)

        return rval


def stitch_rbms(batch_size, rbm_list, niter, inference_procedure=None,
                targets=False):
    """
    Returns a DBM initialized with pre-trained RBMs, with weights and biases
    initialized according to R. Salakhutdinov's policy.

    This method assumes the RBMs were trained normally. It divides the first
    and last hidden layer's weights by two and initialized a hidden layer's
    biases as the mean of its biases and the biases of the visible layer of the
    RBM above it.
    """
    assert len(rbm_list) > 1

    # For intermediary hidden layers, there are two set of biases to choose
    # from: those from the hidden layer of the given RBM, and those from
    # the visible layer of the RBM above it. As in R. Salakhutdinov's code,
    # we handle this by computing the mean of those two sets of biases.
    for this_rbm, above_rbm in zip(rbm_list[:-1], rbm_list[1:]):
        hidden_layer = this_rbm.hidden_layers[0]
        visible_layer = above_rbm.visible_layer
        new_biases = 0.5 * (hidden_layer.get_biases() +
                            visible_layer.get_biases())
        hidden_layer.set_biases(new_biases)

    visible_layer = rbm_list[0].visible_layer
    visible_layer.dbm = None

    hidden_layers = []

    for rbm in rbm_list:
        # Make sure all DBM have only one hidden layer, except for the last
        # one, which can have an optional target layer
        if rbm == rbm_list[-1]:
            if targets:
                assert len(rbm.hidden_layers) == 2
            else:
                assert len(rbm.hidden_layers) == 1
        else:
            assert len(rbm.hidden_layers) == 1

        hidden_layers = hidden_layers + rbm.hidden_layers

    for hidden_layer in hidden_layers:
        hidden_layer.dbm = None

    # Divide first and last hidden layer's weights by two, as described
    # in R. Salakhutdinov's paper (equivalent to training with RBMs with
    # doubled weights)
    first_hidden_layer = hidden_layers[-1]
    if targets:
        last_hidden_layer = hidden_layers[-2]
    else:
        last_hidden_layer = hidden_layers[-1]
    first_hidden_layer.set_weights(0.5 * first_hidden_layer.get_weights())
    last_hidden_layer.set_weights(0.5 * last_hidden_layer.get_weights())

    return DBM(batch_size, visible_layer, hidden_layers, niter,
               inference_procedure)


def flatten(l):
    """
    Turns a nested graph of lists/tuples/other objects
    into a list of objects.
    """
    if isinstance(l, (list, tuple)):
        rval = []
        for elem in l:
            if isinstance(elem, (list, tuple)):
                rval.extend(flatten(elem))
            else:
                rval.append(elem)
    else:
        return [l]
    return rval

def block(l):
    new = []
    for elem in l:
        if isinstance(elem, (list, tuple)):
            new.append(block(elem))
        else:
            new.append(block_gradient(elem))
    if isinstance(l, tuple):
        return tuple(new)
    return new


class GaussianVisLayer(VisibleLayer):
    def __init__(self,
            rows = None,
            cols = None,
            learn_init_inpainting_state=True,
            channels = None,
            nvis = None,
            init_beta = 1.,
            min_beta = 1.,
            init_mu = None,
            tie_beta = None,
            tie_mu = None,
            bias_from_marginals = None,
            beta_lr_scale = 'by_sharing',
            axes = ('b', 0, 1, 'c')):
        """
            Implements a visible layer that is conditionally gaussian with
            diagonal variance. The layer lives in a Conv2DSpace.

            rows, cols, channels: the shape of the space

            init_beta: the initial value of the precision parameter
            min_beta: clip beta so it is at least this big (default 1)

            init_mu: the initial value of the mean parameter

            tie_beta: None or a string specifying how to tie beta
                      'locations' = tie beta across locations, ie
                                    beta should be a vector with one
                                    elem per channel
            tie_mu: None or a string specifying how to tie mu
                    'locations' = tie mu across locations, ie
                                  mu should be a vector with one
                                  elem per channel

        """

        warnings.warn("GaussianVisLayer math very faith based, need to finish working through gaussian.lyx")

        self.__dict__.update(locals())
        del self.self

        if bias_from_marginals is not None:
            del self.bias_from_marginals
            if self.nvis is None:
                raise NotImplementedError()
            assert init_mu is None
            init_mu = bias_from_marginals.X.mean(axis=0)

        if init_mu is None:
            init_mu = 0.

        if nvis is None:
            assert rows is not None
            assert cols is not None
            assert channels is not None
            self.space = Conv2DSpace(shape=[rows,cols], num_channels=channels, axes=axes)
        else:
            assert rows is None
            assert cols is None
            assert channels is None
            self.space = VectorSpace(nvis)
        self.input_space = self.space

        origin = self.space.get_origin()

        beta_origin = origin.copy()
        assert tie_beta in [ None, 'locations']
        if tie_beta == 'locations':
            assert nvis is None
            beta_origin = np.zeros((self.space.num_channels,))
        self.beta = sharedX(beta_origin + init_beta,name = 'beta')
        assert self.beta.ndim == beta_origin.ndim

        mu_origin = origin.copy()
        assert tie_mu in [None, 'locations']
        if tie_mu == 'locations':
            assert nvis is None
            mu_origin = np.zeros((self.space.num_channels,))
        self.mu = sharedX( mu_origin + init_mu, name = 'mu')
        assert self.mu.ndim == mu_origin.ndim

    def get_monitoring_channels(self):
        rval = OrderedDict()

        rval['beta_min'] = self.beta.min()
        rval['beta_mean'] = self.beta.mean()
        rval['beta_max'] = self.beta.max()

        return rval


    def get_params(self):
        if self.mu is None:
            return [self.beta]
        return [self.beta, self.mu]

    def get_lr_scalers(self):
        rval = OrderedDict()

        if self.nvis is None:
            rows, cols = self.space.shape
            num_loc = float(rows * cols)

        assert self.tie_beta in [None, 'locations']
        if self.beta_lr_scale == 'by_sharing':
            if self.tie_beta == 'locations':
                assert self.nvis is None
                rval[self.beta] = 1. / num_loc
        elif self.beta_lr_scale == None:
            pass
        else:
            rval[self.beta] = self.beta_lr_scale

        assert self.tie_mu in [None, 'locations']
        if self.tie_mu == 'locations':
            warn = True
            assert self.nvis is None
            rval[self.mu] = 1./num_loc
            warnings.warn("mu lr_scaler hardcoded to 1/sharing")

        return rval

    def censor_updates(self, updates):
        if self.beta in updates:
            updated_beta = updates[self.beta]
            # updated_beta = Print('updating beta',attrs=['min', 'max'])(updated_beta)
            updates[self.beta] = T.clip(updated_beta,
                    self.min_beta,1e6)




    def broadcasted_mu(self):
        """
        Returns mu, broadcasted to have the same shape as a batch of data
        """

        if self.tie_mu == 'locations':
            def f(x):
                if x == 'c':
                    return 0
                return 'x'
            axes = [f(ax) for ax in self.axes]
            rval = self.mu.dimshuffle(*axes)
        else:
            assert self.tie_mu is None
            if self.nvis is None:
                axes = [0, 1, 2]
                axes.insert(self.axes.index('b'), 'x')
                rval = self.mu.dimshuffle(*axes)
            else:
                rval = self.mu.dimshuffle('x', 0)

        self.input_space.validate(rval)

        return rval

    def broadcasted_beta(self):
        """
        Returns beta, broadcasted to have the same shape as a batch of data
        """
        return self.broadcast_beta(self.beta)

    def broadcast_beta(self, beta):
        """
        Returns beta, broadcasted to have the same shape as a batch of data
        """

        if self.tie_beta == 'locations':
            def f(x):
                if x == 'c':
                    return 0
                return 'x'
            axes = [f(ax) for ax in self.axes]
            rval = beta.dimshuffle(*axes)
        else:
            assert self.tie_beta is None
            if self.nvis is None:
                axes = [0, 1, 2]
                axes.insert(self.axes.index('b'), 'x')
                rval = beta.dimshuffle(*axes)
            else:
                rval = beta.dimshuffle('x', 0)

        self.input_space.validate(rval)

        return rval

    def init_inpainting_state(self, V, drop_mask, noise = False, return_unmasked = False):

        """for Vv, drop_mask_v in get_debug_values(V, drop_mask):
            assert Vv.ndim == 4
            assert drop_mask_v.ndim in [3,4]
            for i in xrange(drop_mask.ndim):
                if Vv.shape[i] != drop_mask_v.shape[i]:
                    print Vv.shape
                    print drop_mask_v.shape
                    assert False
        """

        unmasked = self.broadcasted_mu()

        if drop_mask is None:
            assert not noise
            assert not return_unmasked
            return unmasked
        masked_mu = unmasked * drop_mask
        if not hasattr(self, 'learn_init_inpainting_state'):
            self.learn_init_inpainting_state = True
        if not self.learn_init_inpainting_state:
            masked_mu = block_gradient(masked_mu)
        masked_mu.name = 'masked_mu'

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            unmasked = theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mu.shape,
                    dtype = masked_mu.dtype)
            masked_mu = unmasked * drop_mask
            masked_mu.name = 'masked_noise'


        masked_V  = V  * (1-drop_mask)
        rval = masked_mu + masked_V
        rval.name = 'init_inpainting_state'

        if return_unmasked:
            return rval, unmasked
        return rval


    def expected_energy_term(self, state, average, state_below = None, average_below = None):
        raise NotImplementedError("need to support axes")
        raise NotImplementedError("wasn't implemeneted before axes either")
        assert state_below is None
        assert average_below is None
        self.space.validate(state)
        if average:
            raise NotImplementedError(str(type(self))+" doesn't support integrating out variational parameters yet.")
        else:
            if self.nvis is None:
                axis = (1,2,3)
            else:
                axis = 1
            rval =  0.5 * (self.beta * T.sqr(state - self.mu)).sum(axis=axis)
        assert rval.ndim == 1
        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask = None, V = None,
                        return_unmasked = False):

        msg = layer_above.downward_message(state_above)
        mu = self.broadcasted_mu()

        z = msg + mu
        z.name = 'inpainting_z_[unknown_iter]'

        if drop_mask is not None:
            rval = drop_mask * z + (1-drop_mask) * V
        else:
            rval = z


        rval.name = 'inpainted_V[unknown_iter]'

        if return_unmasked:
            return rval, z

        return rval

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        raise NotImplementedError("need to support axes")

        assert state_below is None
        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu

        rval = theano_rng.normal(size = z.shape, avg = z, dtype = z.dtype,
                       std = 1. / T.sqrt(self.beta) )

        return rval

    def recons_cost(self, V, V_hat_unmasked, drop_mask = None, use_sum=False):

        return self._recons_cost(V=V, V_hat_unmasked=V_hat_unmasked, drop_mask=drop_mask, use_sum=use_sum, beta=self.beta)


    def _recons_cost(self, V, V_hat_unmasked, beta, drop_mask=None, use_sum=False):
        V_hat = V_hat_unmasked

        assert V.ndim == V_hat.ndim
        beta = self.broadcasted_beta()
        unmasked_cost = 0.5 * beta * T.sqr(V-V_hat) - 0.5*T.log(beta / (2*np.pi))
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        if use_sum:
            return masked_cost.mean(axis=0).sum()

        return masked_cost.mean()

        return masked_cost.mean()

    def upward_state(self, total_state):
        if self.nvis is None and total_state.ndim != 4:
            raise ValueError("total_state should have 4 dimensions, has "+str(total_state.ndim))
        assert total_state is not None
        V = total_state
        self.input_space.validate(V)
        upward_state = (V - self.broadcasted_mu()) * self.broadcasted_beta()
        return upward_state

    def make_state(self, num_examples, numpy_rng):
        raise NotImplementedError("need to support axes")

        shape = [num_examples]

        if self.nvis is None:
            rows, cols = self.space.shape
            channels = self.space.num_channels
            shape.append(rows)
            shape.append(cols)
            shape.append(channels)
        else:
            shape.append(self.nvis)

        sample = numpy_rng.randn(*shape)

        sample *= 1./np.sqrt(self.beta.get_value())
        sample += self.mu.get_value()

        rval = sharedX(sample, name = 'v_sample_shared')

        return rval

    def install_presynaptic_outputs(self, outputs_dict, batch_size):

        outputs_dict['output_V_weighted_pred_sum'] = self.space.make_shared_batch(batch_size)

    def ensemble_prediction(self, symbolic, outputs_dict, ensemble):
        """
        Output a symbolic expression for V_hat_unmasked based on taking the
        geometric mean over the ensemble and renormalizing.
        n - 1 members of the ensemble have modified outputs_dict and the nth
        gives its prediction in "symbolic". The parameters for the nth one
        are currently loaded in the model.
        """

        weighted_pred_sum = outputs_dict['output_V_weighted_pred_sum'] \
                + self.broadcasted_beta() * symbolic

        beta_sum = sum(ensemble.get_ensemble_variants(self.beta))

        unmasked_V_hat = weighted_pred_sum / self.broadcast_beta(beta_sum)

        return unmasked_V_hat

    def ensemble_recons_cost(self, V, V_hat_unmasked, drop_mask=None,
            use_sum=False, ensemble=None):

        beta = sum(ensemble.get_ensemble_variants(self.beta)) / ensemble.num_copies

        return self._recons_cost(V=V, V_hat_unmasked=V_hat_unmasked, beta=beta, drop_mask=drop_mask,
            use_sum=use_sum)

class ConvMaxPool(HiddenLayer):
    def __init__(self,
             output_channels,
            kernel_rows,
            kernel_cols,
            pool_rows,
            pool_cols,
            layer_name,
            center = False,
            irange = None,
            sparse_init = None,
            scale_by_sharing = True,
            init_bias = 0.,
            border_mode = 'valid',
            output_axes = ('b', 'c', 0, 1)):
        """


        """
        self.__dict__.update(locals())
        del self.self

        assert (irange is None) != (sparse_init is None)

        self.b = sharedX( np.zeros((output_channels,)) + init_bias, name = layer_name + '_b')
        assert border_mode in ['full','valid']

    def broadcasted_bias(self):

        assert self.b.ndim == 1

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)


    def get_total_state_space(self):
        return CompositeSpace((self.h_space, self.output_space))

    def set_input_space(self, space):
        """ Note: this resets parameters!"""
        if not isinstance(space, Conv2DSpace):
            raise TypeError("ConvMaxPool can only act on a Conv2DSpace, but received " +
                    str(type(space))+" as input.")
        self.input_space = space
        self.input_rows, self.input_cols = space.shape
        self.input_channels = space.num_channels

        if self.border_mode == 'valid':
            self.h_rows = self.input_rows - self.kernel_rows + 1
            self.h_cols = self.input_cols - self.kernel_cols + 1
        else:
            assert self.border_mode == 'full'
            self.h_rows = self.input_rows + self.kernel_rows - 1
            self.h_cols = self.input_cols + self.kernel_cols - 1


        if not( self.h_rows % self.pool_rows == 0):
            raise ValueError("h_rows = %d, pool_rows = %d. Should be divisible but remainder is %d" %
                    (self.h_rows, self.pool_rows, self.h_rows % self.pool_rows))
        assert self.h_cols % self.pool_cols == 0

        self.h_space = Conv2DSpace(shape = (self.h_rows, self.h_cols), num_channels = self.output_channels,
                axes = self.output_axes)
        self.output_space = Conv2DSpace(shape = (self.h_rows / self.pool_rows,
                                                self.h_cols / self.pool_cols),
                                                num_channels = self.output_channels,
                axes = self.output_axes)

        print self.layer_name,': detector shape:',self.h_space.shape,'pool shape:',self.output_space.shape

        if tuple(self.output_axes) == ('b', 0, 1, 'c'):
            self.max_pool = max_pool_b01c
        elif tuple(self.output_axes) == ('b', 'c', 0, 1):
            self.max_pool = max_pool
        else:
            raise NotImplementedError()

        if self.irange is not None:
            self.transformer = make_random_conv2D(self.irange, input_space = space,
                    output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                    batch_size = self.dbm.batch_size, border_mode = self.border_mode, rng = self.dbm.rng)
        else:
            self.transformer = make_sparse_random_conv2D(self.sparse_init, input_space = space,
                    output_space = self.h_space, kernel_shape = (self.kernel_rows, self.kernel_cols),
                    batch_size = self.dbm.batch_size, border_mode = self.border_mode, rng = self.dbm.rng)
        self.transformer._filters.name = self.layer_name + '_W'


        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.center:
            p_ofs, h_ofs = self.init_mf_state()
            self.p_offset = sharedX(self.output_space.get_origin(), 'p_offset')
            self.h_offset = sharedX(self.h_space.get_origin(), 'h_offset')
            f = function([], updates={self.p_offset: p_ofs[0,:,:,:], self.h_offset: h_ofs[0,:,:,:]})
            f()


    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None

        return [ W, self.b]

    def state_to_b01c(self, state):

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_range_rewards(self, state, coeffs):
        """
        TODO: WRITEME
        """
        rval = 0.

        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            if c == 0.:
                continue
            # Range over everything but the channel index
            # theano can only take gradient through max if the max is over 1 axis or all axes
            # so I manually unroll the max for the case I use here
            assert self.h_space.axes == ('b', 'c', 0, 1)
            assert self.output_space.axes == ('b', 'c', 0, 1)
            mx = s.max(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mx.owner.op, 'grad')
            mn = s.min(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mn.owner.op, 'grad')
            assert mx.ndim == 1
            assert mn.ndim == 1
            r = mx - mn
            rval += (1. - r).mean() * c

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps):
        """

            target: if pools contain more than one element, should be a list with
                    two elements. the first element is for the pooling units and
                    the second for the detector units.

        """
        rval = 0.


        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(target, float)
            assert isinstance(coeff, float)
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
            if eps is None:
                eps = 0.
            eps = [eps]
        else:
            if eps is None:
                eps = [0., 0.]
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            p_target, h_target = target
            if h_target > p_target and (coeff[0] != 0. and coeff[1] != 0.):
                # note that, within each group, E[p] is the sum of E[h]
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            if c == 0.:
                continue
            # Average over everything but the channel index
            m = s.mean(axis= [ ax for ax in range(4) if self.output_axes[ax] != 'c' ])
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_lr_scalers(self):
        if self.scale_by_sharing:
            # scale each learning rate by 1 / # times param is reused
            h_rows, h_cols = self.h_space.shape
            num_h = float(h_rows * h_cols)
            return OrderedDict([(self.transformer._filters, 1./num_h),
                     (self.b, 1. / num_h)])
        else:
            return OrderedDict()

    def upward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return p

    def downward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return h

    def get_monitoring_channels_from_state(self, state):

        P, H = state

        if tuple(self.output_axes) == ('b',0,1,'c'):
            p_max = P.max(axis=(0,1,2))
            p_min = P.min(axis=(0,1,2))
            p_mean = P.mean(axis=(0,1,2))
        else:
            assert tuple(self.output_axes) == ('b','c',0,1)
            p_max = P.max(axis=(0,2,3))
            p_min = P.min(axis=(0,2,3))
            p_mean = P.mean(axis=(0,2,3))
        p_range = p_max - p_min

        rval = {
                'p_max_max' : p_max.max(),
                'p_max_mean' : p_max.mean(),
                'p_max_min' : p_max.min(),
                'p_min_max' : p_min.max(),
                'p_min_mean' : p_min.mean(),
                'p_min_max' : p_min.max(),
                'p_range_max' : p_range.max(),
                'p_range_mean' : p_range.mean(),
                'p_range_min' : p_range.min(),
                'p_mean_max' : p_mean.max(),
                'p_mean_mean' : p_mean.mean(),
                'p_mean_min' : p_mean.min()
                }

        return rval

    def get_weight_decay(self, coeffs):
        W , = self.transformer.get_params()
        return coeffs * T.sqr(W).sum()



    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if not hasattr(state_below, 'ndim'):
            raise TypeError("state_below should be a TensorType, got " +
                    str(state_below) + " of type " + str(type(state_below)))
        if state_below.ndim != 4:
            raise ValueError("state_below should have ndim 4, has "+str(state_below.ndim))

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = self.max_pool(z, (self.pool_rows, self.pool_cols), msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
            try:
                self.output_space.validate(msg)
            except TypeError, e:
                raise TypeError(str(type(layer_above))+".downward_message gave something that was not the right type: "+str(e))
        else:
            msg = None

        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        p, h, p_sample, h_sample = self.max_pool(z,
                (self.pool_rows, self.pool_cols), msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        self.h_space.validate(downward_state)
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw,(outp,rows,cols,inp))


    def init_mf_state(self):
        default_z = self.broadcasted_bias()
        shape = {
                'b': self.dbm.batch_size,
                0: self.h_space.shape[0],
                1: self.h_space.shape[1],
                'c': self.h_space.num_channels
                }
        # work around theano bug with broadcasted stuff
        default_z += T.alloc(*([0.]+[shape[elem] for elem in self.h_space.axes])).astype(default_z.dtype)
        assert default_z.ndim == 4

        p, h = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols))

        return p, h

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(self.dbm.batch_size)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.broadcasted_bias()

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        p_exp, h_exp, p_sample, h_sample = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols),
                theano_rng = theano_rng)

        p_state = sharedX( self.output_space.get_origin_batch(
            self.dbm.batch_size))


        t2 = time.time()

        f = function([], updates = [
            (p_state, p_sample),
            (h_state, h_sample)
            ])

        t3 = time.time()

        f()

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):

        self.input_space.validate(state_below)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = (downward_state * self.broadcasted_bias()).sum(axis=(1,2,3))
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=(1,2,3))

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval

class ConvC01B_MaxPool(HiddenLayer):
    def __init__(self,
             output_channels,
            kernel_shape,
            pool_rows,
            pool_cols,
            layer_name,
            center = False,
            irange = None,
            sparse_init = None,
            scale_by_sharing = True,
            init_bias = 0.,
            pad = 0,
            partial_sum = 1):
        """
        Like ConvMaxPool but using cuda convnet for the backend.

        kernel_shape: two-element list or tuple of ints specifying
                    rows and columns of kernel
                    currently the two must be the same
        output_channels: the number of convolutional channels in the
            output and pooling layer.
        """
        self.__dict__.update(locals())
        del self.self

        assert (irange is None) != (sparse_init is None)
        self.output_axes = ('c', 0, 1, 'b')
        self.detector_channels = output_channels
        self.tied_b = 1

    def broadcasted_bias(self):

        if self.b.ndim != 1:
            raise NotImplementedError()

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)


    def get_total_state_space(self):
        return CompositeSpace((self.h_space, self.output_space))

    def set_input_space(self, space):
        """ Note: this resets parameters!"""

        setup_detector_layer_c01b(layer=self,
                input_space=space, rng=self.dbm.rng,
                irange=self.irange)

        if not tuple(space.axes) == ('c', 0, 1, 'b'):
            raise AssertionError("You're not using c01b inputs. Ian is enforcing c01b inputs while developing his pipeline to make sure it runs at maximal speed. If you really don't want to use c01b inputs, you can remove this check and things should work. If they don't work it's only because they're not tested.")
        if self.dummy_channels != 0:
            raise NotImplementedError(str(type(self))+" does not support adding dummy channels for cuda-convnet compatibility yet, you must implement that feature or use inputs with <=3 channels or a multiple of 4 channels")

        self.input_rows = self.input_space.shape[0]
        self.input_cols = self.input_space.shape[1]
        self.h_rows = self.detector_space.shape[0]
        self.h_cols = self.detector_space.shape[1]

        if not(self.h_rows % self.pool_rows == 0):
            raise ValueError(self.layer_name + ": h_rows = %d, pool_rows = %d. Should be divisible but remainder is %d" %
                    (self.h_rows, self.pool_rows, self.h_rows % self.pool_rows))
        assert self.h_cols % self.pool_cols == 0

        self.h_space = Conv2DSpace(shape = (self.h_rows, self.h_cols), num_channels = self.output_channels,
                axes = self.output_axes)
        self.output_space = Conv2DSpace(shape = (self.h_rows / self.pool_rows,
                                                self.h_cols / self.pool_cols),
                                                num_channels = self.output_channels,
                axes = self.output_axes)

        print self.layer_name,': detector shape:',self.h_space.shape,'pool shape:',self.output_space.shape

        assert tuple(self.output_axes) == ('c', 0, 1, 'b')
        self.max_pool = max_pool_c01b

        if self.center:
            p_ofs, h_ofs = self.init_mf_state()
            self.p_offset = sharedX(self.output_space.get_origin(), 'p_offset')
            self.h_offset = sharedX(self.h_space.get_origin(), 'h_offset')
            f = function([], updates={self.p_offset: p_ofs[:,:,:,0], self.h_offset: h_ofs[:,:,:,0]})
            f()


    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None

        return [ W, self.b]

    def state_to_b01c(self, state):

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_range_rewards(self, state, coeffs):
        """
        TODO: WRITEME
        """
        rval = 0.

        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(coeffs, float)
            _, state = state
            state = [state]
            coeffs = [coeffs]
        else:
            assert all([len(elem) == 2 for elem in [state, coeffs]])

        for s, c in safe_zip(state, coeffs):
            if c == 0.:
                continue
            # Range over everything but the channel index
            # theano can only take gradient through max if the max is over 1 axis or all axes
            # so I manually unroll the max for the case I use here
            assert self.h_space.axes == ('b', 'c', 0, 1)
            assert self.output_space.axes == ('b', 'c', 0, 1)
            mx = s.max(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mx.owner.op, 'grad')
            mn = s.min(axis=3).max(axis=2).max(axis=0)
            assert hasattr(mn.owner.op, 'grad')
            assert mx.ndim == 1
            assert mn.ndim == 1
            r = mx - mn
            rval += (1. - r).mean() * c

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps):
        """

            target: if pools contain more than one element, should be a list with
                    two elements. the first element is for the pooling units and
                    the second for the detector units.

        """
        rval = 0.


        if self.pool_rows == 1 and self.pool_cols == 1:
            # If the pool size is 1 then pools = detectors
            # and we should not penalize pools and detectors separately
            assert len(state) == 2
            assert isinstance(target, float)
            assert isinstance(coeff, float)
            _, state = state
            state = [state]
            target = [target]
            coeff = [coeff]
            if eps is None:
                eps = 0.
            eps = [eps]
        else:
            if eps is None:
                eps = [0., 0.]
            assert all([len(elem) == 2 for elem in [state, target, coeff]])
            p_target, h_target = target
            if h_target > p_target and (coeff[0] != 0. and coeff[1] != 0.):
                # note that, within each group, E[p] is the sum of E[h]
                warnings.warn("Do you really want to regularize the detector units to be more active than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            if c == 0.:
                continue
            # Average over everything but the channel index
            m = s.mean(axis= [ ax for ax in range(4) if self.output_axes[ax] != 'c' ])
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.scale_by_sharing:
            # scale each learning rate by 1 / # times param is reused
            h_rows, h_cols = self.h_space.shape
            num_h = float(h_rows * h_cols)
            rval[self.transformer._filters] = 1. /num_h
            rval[self.b] = 1. / num_h

        return rval

    def upward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return p

    def downward_state(self, total_state):
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return h

    def get_monitoring_channels_from_state(self, state):

        P, H = state

        axes = tuple([i for i, ax in enumerate(self.output_axes) if ax != 'c'])
        p_max = P.max(axis=(0,1,2))
        p_min = P.min(axis=(0,1,2))
        p_mean = P.mean(axis=(0,1,2))

        p_range = p_max - p_min

        rval = {
                'p_max_max' : p_max.max(),
                'p_max_mean' : p_max.mean(),
                'p_max_min' : p_max.min(),
                'p_min_max' : p_min.max(),
                'p_min_mean' : p_min.mean(),
                'p_min_max' : p_min.max(),
                'p_range_max' : p_range.max(),
                'p_range_mean' : p_range.mean(),
                'p_range_min' : p_range.min(),
                'p_mean_max' : p_mean.max(),
                'p_mean_mean' : p_mean.mean(),
                'p_mean_min' : p_mean.min()
                }

        return rval

    def get_weight_decay(self, coeffs):
        W , = self.transformer.get_params()
        return coeffs * T.sqr(W).sum()

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if not hasattr(state_below, 'ndim'):
            raise TypeError("state_below should be a TensorType, got " +
                    str(state_below) + " of type " + str(type(state_below)))
        if state_below.ndim != 4:
            raise ValueError("state_below should have ndim 4, has "+str(state_below.ndim))

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = self.max_pool(z, (self.pool_rows, self.pool_cols), msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        raise NotImplementedError("Need to update for C01B")

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
            try:
                self.output_space.validate(msg)
            except TypeError, e:
                raise TypeError(str(type(layer_above))+".downward_message gave something that was not the right type: "+str(e))
        else:
            msg = None

        z = self.transformer.lmul(state_below) + self.broadcasted_bias()
        p, h, p_sample, h_sample = self.max_pool(z,
                (self.pool_rows, self.pool_cols), msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        self.h_space.validate(downward_state)
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    def init_mf_state(self):
        default_z = self.broadcasted_bias()
        shape = {
                'b': self.dbm.batch_size,
                0: self.h_space.shape[0],
                1: self.h_space.shape[1],
                'c': self.h_space.num_channels
                }
        # work around theano bug with broadcasted stuff
        default_z += T.alloc(*([0.]+[shape[elem] for elem in self.h_space.axes])).astype(default_z.dtype)
        assert default_z.ndim == 4

        p, h = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols))

        return p, h

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """
        raise NotImplementedError("Need to update for C01B")

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(self.dbm.batch_size)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.broadcasted_bias()

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        p_exp, h_exp, p_sample, h_sample = self.max_pool(
                z = default_z,
                pool_shape = (self.pool_rows, self.pool_cols),
                theano_rng = theano_rng)

        p_state = sharedX( self.output_space.get_origin_batch(
            self.dbm.batch_size))


        t2 = time.time()

        f = function([], updates = [
            (p_state, p_sample),
            (h_state, h_sample)
            ])

        t3 = time.time()

        f()

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):

        raise NotImplementedError("Need to update for C01B")
        self.input_space.validate(state_below)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = (downward_state * self.broadcasted_bias()).sum(axis=(1,2,3))
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=(1,2,3))

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval

class BVMP_Gaussian(BinaryVectorMaxPool):
    """
    Like BinaryVectorMaxPool, but must have GaussianVisLayer
    as its input. Uses its beta to bias the hidden units appropriately.
    See gaussian.lyx

    beta is *not* considered a parameter of this layer, it's just an
    external factor influencing how this layer behaves.
    Gradient can still flow to beta, but it will only be included in
    the parameters list if some class other than this layer includes it.
    """

    def __init__(self,
            input_layer,
             detector_layer_dim,
            pool_size,
            layer_name,
            irange = None,
            sparse_init = None,
            sparse_stdev = 1.,
            include_prob = 1.0,
            init_bias = 0.,
            W_lr_scale = None,
            b_lr_scale = None,
            center = False,
            mask_weights = None,
            max_col_norm = None,
            copies = 1):
        """

            include_prob: probability of including a weight element in the set
                    of weights initialized to U(-irange, irange). If not included
                    it is initialized to 0.

        """

        warnings.warn("BVMP_Gaussian math is very faith-based, need to complete gaussian.lyx")

        args = locals()

        del args['input_layer']
        del args['self']
        super(BVMP_Gaussian, self).__init__(**args)
        self.input_layer = input_layer

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        x = raw_input("multiply by beta?")
        if x == 'y':
            beta = self.input_layer.beta.get_value()
            return (W.T * beta).T
        assert x == 'n'
        return W

    def set_weights(self, weights):
        raise NotImplementedError("beta would make get_weights for visualization not correspond to set_weights")
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter = False):
        self.b.set_value(biases)
        if recenter:
            assert self.center
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset.set_value(sigmoid_numpy(self.b.get_value()))

    def get_biases(self):
        return self.b.get_value() - self.beta_bias().eval()


    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        raise NotImplementedError("need to account for beta")
        if self.copies != 1:
            raise NotImplementedError()

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to None so that it may appear after layer_above / state_above in the list.")

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
        else:
            msg = None

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b
        p, h, p_sample, h_sample = max_pool_channels(z,
                self.pool_size, msg, theano_rng)

        return p_sample, h_sample

    def downward_message(self, downward_state):
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval * self.copies

    def init_mf_state(self):
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size, self.detector_layer_dim).astype(self.b.dtype) + \
                self.b.dimshuffle('x', 0) + self.beta_bias()
        rval = max_pool_channels(z = z,
                pool_size = self.pool_size)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """
        raise NotImplementedError("need to account for beta")

        if not hasattr(self, 'copies'):
            self.copies = 1

        if self.copies != 1:
            raise NotImplementedError()


        empty_input = self.h_space.get_origin_batch(num_examples)
        empty_output = self.output_space.get_origin_batch(num_examples)

        h_state = sharedX(empty_input)
        p_state = sharedX(empty_output)

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        default_z = T.zeros_like(h_state) + self.b

        p_exp, h_exp, p_sample, h_sample = max_pool_channels(
                z = default_z,
                pool_size = self.pool_size,
                theano_rng = theano_rng)

        assert h_sample.dtype == default_z.dtype

        f = function([], updates = [
            (p_state , p_sample),
            (h_state , h_sample)
            ])

        f()

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):
        raise NotImplementedError("need to account for beta, and maybe some oether stuff")

        # Don't need to do anything special for centering, upward_state / downward state
        # make it all just work

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        downward_state = self.downward_state(state)
        self.h_space.validate(downward_state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        # Specifically, our terms are -u^T W d - b^T d where u is the upward state of layer below
        # and d is the downward state of this layer

        bias_term = T.dot(downward_state, self.b)
        weights_term = (self.transformer.lmul(state_below) * downward_state).sum(axis=1)

        rval = -bias_term - weights_term

        assert rval.ndim == 1

        return rval * self.copies

    def linear_feed_forward_approximation(self, state_below):
        """
        Used to implement TorontoSparsity. Unclear exactly what properties of it are
        important or how to implement it for other layers.

        Properties it must have:
            output is same kind of data structure (ie, tuple of theano 2-tensors)
            as mf_update

        Properties it probably should have for other layer types:
            An infinitesimal change in state_below or the parameters should cause the same sign of change
            in the output of linear_feed_forward_approximation and in mf_update

            Should not have any non-linearities that cause the gradient to shrink

            Should disregard top-down feedback
        """
        raise NotImplementedError("need to account for beta")

        z = self.transformer.lmul(state_below) + self.b

        if self.pool_size != 1:
            # Should probably implement sum pooling for the non-pooled version,
            # but in reality it's not totally clear what the right answer is
            raise NotImplementedError()

        return z, z

    def beta_bias(self):
        W, = self.transformer.get_params()
        beta = self.input_layer.beta
        assert beta.ndim == 1
        return - 0.5 * T.dot(beta, T.sqr(W))

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_'+layer_above.layer_name+'_to_'+self.layer_name+'['+iter_name+']'
        else:
            msg = None

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.b + self.beta_bias()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        p,h = max_pool_channels(z, self.pool_size, msg)

        p.name = self.layer_name + '_p_' + iter_name
        h.name = self.layer_name + '_h_' + iter_name

        return p, h


class SuperWeightDoubling(WeightDoubling):

    def multi_infer(self, V, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert return_history in [True, False, 0, 1]

        if niter is None:
            niter = dbm.niter

        new_V = 0.5 * V + 0.5 * dbm.visible_layer.init_inpainting_state(V,drop_mask = None,noise = False, return_unmasked = False)

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.visible_layer.upward_state(new_V),
                                                            iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                                                            iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.visible_layer.upward_state(V)))

        if block_grad == 1:
            H_hat = block(H_hat)

        history = [ (new_V, list(H_hat)) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(1, niter):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = dbm.visible_layer.upward_state(new_V)
                    else:
                        state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                V_hat = dbm.visible_layer.inpaint_update(
                                                                                 state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                                                                                 layer_above = dbm.hidden_layers[0],
                                                                                 V = V,
                                                                                 drop_mask = None)
                new_V = 0.5 * V_hat + 0.5 * V

                for j in xrange(1,len(H_hat),2):
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                #end ifelse
                #end for odd layer

                if block_grad == i:
                    H_hat = block(H_hat)
                    V_hat = block_gradient(V_hat)

                history.append((new_V, list(H_hat)))
        # end for mf iter
        # end if recurrent
        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        inferred = H_hat
        for elem in flatten(inferred):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            assert V in gof.graph.ancestors([elem])

        if return_history:
            return history
        else:
            return H_hat[-1]

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow, Mehdi Mirza,
            Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:
            unsupervised:
                Y and drop_mask_Y are not passed to the method.
                The method produces V_hat, an inpainted version of V
            supervised:
                Y and drop_mask_Y are passed to the method.
                The method produces V_hat and Y_hat

        V: a theano batch in model.input_space
        Y: a theano batch in model.output_space, ie, in the output
            space of the last hidden layer
            (it's not really a hidden layer anymore, but oh well.
            it's convenient to code it this way because the labels
            are sort of "on top" of everything else)
            *** Y is always assumed to be a matrix of one-hot category
            labels. ***
        drop_mask: a theano batch in model.input_space
            Should be all binary, with 1s indicating that the corresponding
            element of X should be "dropped", ie, hidden from the algorithm
            and filled in as part of the inpainting process
        drop_mask_Y: a theano vector
            Since we assume Y is a one-hot matrix, each row is a single
            categorical variable. drop_mask_Y is a binary mask specifying
            which *rows* to drop.
        """

        dbm = self.dbm

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = dbm.niter

        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
    "so each example is only one variable. drop_mask_Y should "
    "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        # Last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                #layer_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            dirty_term = drop_mask_Y * Y_hat_unmasked
            clean_term = (1 - drop_mask_Y) * Y
            Y_hat = dirty_term + clean_term
            H_hat[-1] = Y_hat
            if len(dbm.hidden_layers) > 1:
                i = len(dbm.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : list(H_hat), 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append(d)

        if block_grad == 1:
            V_hat = block_gradient(V_hat)
            V_hat_unmasked = block_gradient(V_hat_unmasked)
            H_hat = block(H_hat)
        update_history()

        for i in xrange(niter-1):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat

class MoreConsistent(SuperWeightDoubling):
    """
    There's an oddity in SuperWeightDoubling where during the inpainting, we
    initialize Y_hat to sigmoid(biases) if a clean Y is passed in and 2 * weights
    otherwise. I believe but ought to check that mf always does weight doubling.
    This class makes the two more consistent by just implementing mf as calling
    inpainting with Y masked out.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        drop_mask = T.zeros_like(V)

        if Y is not None:
            # Y is observed, specify that it's fully observed
            drop_mask_Y = T.zeros_like(Y)
        else:
            # Y is not observed
            last_layer = self.dbm.hidden_layers[-1]
            if isinstance(last_layer, Softmax):
                # Y is not observed, the model has a Y variable, fill in a dummy one
                # and specify that no element of it is observed
                batch_size = self.dbm.get_input_space().batch_size(V)
                num_classes = self.dbm.hidden_layers[-1].n_classes
                assert isinstance(num_classes, int)
                Y = T.alloc(1., batch_size, num_classes)
                drop_mask_Y = T.alloc(1., batch_size)
            else:
                # Y is not observed because the model has no Y variable
                drop_mask_Y = None

        history = self.do_inpainting(V=V,
            Y=Y,
            return_history=True,
            drop_mask=drop_mask,
            drop_mask_Y=drop_mask_Y,
            noise=False,
            niter=niter,
            block_grad=block_grad)

        assert history[-1]['H_hat'][0] is not history[-2]['H_hat'][0] # rm

        if return_history:
            return [elem['H_hat'] for elem in history]

        rval =  history[-1]['H_hat']

        if 'Y_hat_unmasked' in history[-1]:
            rval[-1] = history[-1]['Y_hat_unmasked']

        return rval

class MoreConsistent2(WeightDoubling):

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow, Mehdi Mirza,
            Aaron Courville, and Yoshua Bengio. NIPS 2013.

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:
            unsupervised:
                Y and drop_mask_Y are not passed to the method.
                The method produces V_hat, an inpainted version of V
            supervised:
                Y and drop_mask_Y are passed to the method.
                The method produces V_hat and Y_hat

        V: a theano batch in model.input_space
        Y: a theano batch in model.output_space, ie, in the output
            space of the last hidden layer
            (it's not really a hidden layer anymore, but oh well.
            it's convenient to code it this way because the labels
            are sort of "on top" of everything else)
            *** Y is always assumed to be a matrix of one-hot category
            labels. ***
        drop_mask: a theano batch in model.input_space
            Should be all binary, with 1s indicating that the corresponding
            element of X should be "dropped", ie, hidden from the algorithm
            and filled in as part of the inpainting process
        drop_mask_Y: a theano vector
            Since we assume Y is a one-hot matrix, each row is a single
            categorical variable. drop_mask_Y is a binary mask specifying
            which *rows* to drop.
        """

        dbm = self.dbm

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = dbm.niter

        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
    "so each example is only one variable. drop_mask_Y should "
    "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        # Last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                #layer_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = H_hat[-1]
            dirty_term = drop_mask_Y * Y_hat_unmasked
            clean_term = (1 - drop_mask_Y) * Y
            Y_hat = dirty_term + clean_term
            H_hat[-1] = Y_hat
            """
            if len(dbm.hidden_layers) > 1:
                i = len(dbm.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')
            """


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : list(H_hat), 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append(d)

        if block_grad == 1:
            V_hat = block_gradient(V_hat)
            V_hat_unmasked = block_gradient(V_hat_unmasked)
            H_hat = block(H_hat)
        update_history()

        for i in xrange(niter-1):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat



class BiasInit(InferenceProcedure):
    """
    An InferenceProcedure that initializes the mean field parameters based on the
    biases in the model. This InferenceProcedure uses the same weights at every
    iteration, rather than doubling the weights on the first pass.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            dbm.hidden_layers[-1].get_output_space().validate(Y)

        if niter is None:
            niter = dbm.niter

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [ list(H_hat) ]

        #we only need recurrent inference if there are multiple layers
        assert (niter > 1) == (len(dbm.hidden_layers) > 1)

        for i in xrange(niter):
            for j in xrange(0,len(H_hat),2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)

            if Y is not None:
                H_hat[-1] = Y

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    state_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                #end ifelse
            #end for odd layer

            if Y is not None:
                H_hat[-1] = Y

            for i, elem in enumerate(H_hat):
                if elem is Y:
                    assert i == len(H_hat) -1
                    continue
                else:
                    assert elem not in history[-1]


            if block_grad == i + 1:
                H_hat = block(H_hat)

            history.append(list(H_hat))
        # end for mf iter

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        if Y is not None:
            assert H_hat[-1] is Y
            inferred = H_hat[:-1]
        else:
            inferred = H_hat
        for elem in flatten(inferred):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            if V not in theano.gof.graph.ancestors([elem]):
                print str(elem)+" does not have V as an ancestor!"
                print theano.printing.min_informative_str(V)
                if elem is V:
                    print "this variational parameter *is* V"
                else:
                    print "this variational parameter is not the same as V"
                print "V is ",V
                assert False
            if Y is not None:
                assert Y in theano.gof.graph.ancestors([elem])

        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        for elem in history:
            assert len(elem) == len(dbm.hidden_layers)

        if return_history:
            for hist_elem, H_elem in safe_zip(history[-1], H_hat):
                assert hist_elem is H_elem
            return history
        else:
            return H_hat

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow, Mehdi Mirza,
            Aaron Courville, and Yoshua Bengio. NIPS 2013.
        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:
            unsupervised:
                Y and drop_mask_Y are not passed to the method.
                The method produces V_hat, an inpainted version of V
            supervised:
                Y and drop_mask_Y are passed to the method.
                The method produces V_hat and Y_hat

        V: a theano batch in model.input_space
        Y: a theano batch in model.output_space, ie, in the output
            space of the last hidden layer
            (it's not really a hidden layer anymore, but oh well.
            it's convenient to code it this way because the labels
            are sort of "on top" of everything else)
            *** Y is always assumed to be a matrix of one-hot category
            labels. ***
        drop_mask: a theano batch in model.input_space
            Should be all binary, with 1s indicating that the corresponding
            element of X should be "dropped", ie, hidden from the algorithm
            and filled in as part of the inpainting process
        drop_mask_Y: a theano vector
            Since we assume Y is a one-hot matrix, each row is a single
            categorical variable. drop_mask_Y is a binary mask specifying
            which *rows* to drop.
        """

        dbm = self.dbm

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = dbm.niter


        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append( d )

        update_history()

        for i in xrange(niter):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat

class UpDown(InferenceProcedure):
    """
    An InferenceProcedure that initializes the mean field parameters based on the
    biases in the model, then alternates between updating each of the layers bottom-to-top
    and updating each of the layers top-to-bottom.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):

        dbm = self.dbm

        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            dbm.hidden_layers[-1].get_output_space().validate(Y)

        if niter is None:
            niter = dbm.niter

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [ list(H_hat) ]

        #we only need recurrent inference if there are multiple layers
        assert (niter > 1) == (len(dbm.hidden_layers) > 1)

        for i in xrange(niter):
            # Determine whether to go up or down on this iteration
            if i % 2 == 0:
                start = 0
                stop = len(H_hat)
                inc = 1
            else:
                start = len(H_hat) - 1
                stop = -1
                inc = -1
            # Do the mean field updates
            for j in xrange(start, stop, inc):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None:
                    H_hat[-1] = Y

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    state_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                #end ifelse
            #end for odd layer

            if Y is not None:
                H_hat[-1] = Y

            if block_grad == i + 1:
                H_hat = block(H_hat)

            history.append(list(H_hat))
        # end for mf iter

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)
        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow, Mehdi Mirza,
            Aaron Courville, and Yoshua Bengio. NIPS 2013.

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:
            unsupervised:
                Y and drop_mask_Y are not passed to the method.
                The method produces V_hat, an inpainted version of V
            supervised:
                Y and drop_mask_Y are passed to the method.
                The method produces V_hat and Y_hat

        V: a theano batch in model.input_space
        Y: a theano batch in model.output_space, ie, in the output
            space of the last hidden layer
            (it's not really a hidden layer anymore, but oh well.
            it's convenient to code it this way because the labels
            are sort of "on top" of everything else)
            *** Y is always assumed to be a matrix of one-hot category
            labels. ***
        drop_mask: a theano batch in model.input_space
            Should be all binary, with 1s indicating that the corresponding
            element of X should be "dropped", ie, hidden from the algorithm
            and filled in as part of the inpainting process
        drop_mask_Y: a theano vector
            Since we assume Y is a one-hot matrix, each row is a single
            categorical variable. drop_mask_Y is a binary mask specifying
            which *rows* to drop.
        """

        if Y is not None:
            assert isinstance(self.hidden_layers[-1], Softmax)

        model = self.dbm

        warnings.warn("""Should add unit test that calling this with a batch of
                different inputs should yield the same output for each if noise
                is False and drop_mask is all 1s""")

        if niter is None:
            niter = model.niter


        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(model.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = model.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [None] + [layer.init_mf_state() for layer in model.hidden_layers[1:]]

        if Y is not None:
            Y_hat_unmasked = model.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append( d )

        update_history()

        for i in xrange(niter):

            if i % 2 == 0:
                start = 0
                stop = len(H_hat)
                inc = 1
                if i > 0:
                    # Don't start by updating V_hat on iteration 0 or this will throw out the
                    # noise
                    V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                            state_above = model.hidden_layers[0].downward_state(H_hat[0]),
                            layer_above = model.hidden_layers[0],
                            V = V,
                            drop_mask = drop_mask, return_unmasked = True)
                    V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)
            else:
                start = len(H_hat) - 1
                stop = -1
                inc = -1
            for j in xrange(start, stop, inc):
                if j == 0:
                    state_below = model.visible_layer.upward_state(V_hat)
                else:
                    state_below = model.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = model.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = model.hidden_layers[j+1]
                H_hat[j] = model.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(model.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            if i % 2 == 1:
                V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                        state_above = model.hidden_layers[0].downward_state(H_hat[0]),
                        layer_above = model.hidden_layers[0],
                        V = V,
                        drop_mask = drop_mask, return_unmasked = True)
                V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat
