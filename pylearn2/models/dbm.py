"""
This module contains functionality related to deep Boltzmann machines.
They are implemented generically in order to make it easy to support
convolution versions, etc.

Some of the code needed to actually use a DBM might not be in this
repository yet. Ian is gradually moving pieces of it over from his
private repository. Some of his code contains private research ideas
that he can't move to this repository until he has a paper on them.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np
import sys
import time
import warnings

from theano import config
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import block_gradient
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
        if inference_procedure is not None:
            inference_procedure.set_dbm(self)

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

    def setup_rng(self):
        self.rng = np.random.RandomState([2012, 10, 17])

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

        for param in self.visible_layer.get_params():
            assert param.name is not None
        rval = self.visible_layer.get_params()
        for layer in self.hidden_layers:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
                    assert False
            rval = rval.union(layer.get_params())

        # Patch pickle files that predate the freeze_set feature
        if not hasattr(self, 'freeze_set'):
            self.freeze_set = set([])

        rval = set([elem for elem in rval if elem not in self.freeze_set])

        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.hidden_layers:
            layer.set_batch_size(batch_size)

    def censor_updates(self, updates):
        self.visible_layer.censor_updates(updates)
        for layer in self.hidden_layers:
            layer.censor_updates(updates)

    def get_input_space(self):
        return self.visible_layer.space

    def get_lr_scalers(self):
        rval = {}

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

        states = [ layer.make_state(num_examples, rng) for layer in layers ]

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

        rval = dict(zipped)

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
        assert isinstance(num_steps, int)
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
            layer_to_clamp = {}

        for key in layer_to_clamp:
            assert key is self.visible_layer or key in self.hidden_layers

        for layer in [self.visible_layer] + self.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        #Assemble the return value
        layer_to_updated = {}

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

        rval = {}

        def add_updates(old, new):
            if isinstance(old, (list, tuple)):
                for old_elem, new_elem in safe_izip(old, new):
                    add_updates(old_elem, new_elem)
            else:
                rval[old] = new

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = {}

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

    def get_monitoring_channels(self, X, Y = None):

        history = self.mf(X, return_history = True)
        q = history[-1]

        rval = {}

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

    def get_test_batch_size(self):
        return self.batch_size



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
        return {}

    def get_monitoring_channels_from_state(self, state):
        """
        TODO WRITEME
        """
        return {}

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

    def get_range_rewards(self, state, coess):
        raise NotImplementedError(str(type(self))+" does not implement get_range_rewards")

    def get_l1_act_cost(self, state, target, coeff, eps):
        raise NotImplementedError(str(type(self))+" does not implement get_l1_act_cost")

class BinaryVector(VisibleLayer):
    """
    A DBM visible layer consisting of binary random variables living
    in a VectorSpace.
    """

    def __init__(self,
            nvis,
            bias_from_marginals = None):
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
            X = bias_from_marginals.get_design_matrix()
            assert X.max() == 1.
            assert X.min() == 0.
            assert not np.any( (X > 0.) * (X < 1.) )

            mean = X.mean(axis=0)

            mean = np.clip(mean, 1e-7, 1-1e-7)

            init_bias = inverse_sigmoid_numpy(mean)

        self.bias = sharedX(init_bias, 'visible_bias')

    def get_biases(self):
        return self.bias.get_value()

    def set_biases(self, biases):
        self.bias.set_value(biases)


    def get_params(self):
        return set([self.bias])

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

        assert state_below is None

        msg = layer_above.downward_message(state_above)

        bias = self.bias

        z = msg + bias

        phi = T.nnet.sigmoid(z)

        rval = theano_rng.binomial(size = phi.shape, p = phi, dtype = phi.dtype,
                       n = 1 )

        return rval

    def make_state(self, num_examples, numpy_rng):

        driver = numpy_rng.uniform(0.,1., (num_examples, self.nvis))
        mean = sigmoid_numpy(self.bias.get_value())
        sample = driver < mean

        rval = sharedX(sample, name = 'v_sample_shared')

        return rval

    def expected_energy_term(self, state, average, state_below = None, average_below = None):

        assert state_below is None
        assert average_below is None
        assert average in [True, False]
        self.space.validate(state)

        # Energy function is linear so it doesn't matter if we're averaging or not
        rval = -T.dot(state, self.bias)

        assert rval.ndim == 1

        return rval

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
            mask_weights = None):
        """

            include_prob: probability of including a weight element in the set
                    of weights initialized to U(-irange, irange). If not included
                    it is initialized to 0.

        """
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')


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

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

    def get_total_state_space(self):
        return CompositeSpace((self.output_space, self.h_space))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        return self.transformer.get_params().union([self.b])

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

    def set_biases(self, biases):
        self.b.set_value(biases)

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
        return p

    def downward_state(self, total_state):
        p,h = total_state
        return h

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return {
              'row_norms_min'  : row_norms.min(),
              'row_norms_mean' : row_norms.mean(),
              'row_norms_max'  : row_norms.max(),
              'col_norms_min'  : col_norms.min(),
              'col_norms_mean' : col_norms.mean(),
              'col_norms_max'  : col_norms.max(),
            }


    def get_monitoring_channels_from_state(self, state):

        P, H = state

        rval ={}

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

    def get_range_rewards(self, state, coeffs):
        rval = 0.

        P, H = state
        self.output_space.validate(P)
        self.h_space.validate(H)


        if self.pool_size == 1:
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
            if target[1] < target[0]:
                warnings.warn("Do you really want to regularize the detector units to be sparser than the pooling units?")

        for s, t, c, e in safe_zip(state, target, coeff, eps):
            assert all([isinstance(elem, float) or hasattr(elem, 'dtype') for elem in [t, c, e]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

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

        return rval

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

        f = function([], updates = {
            p_state : p_sample,
            h_state : h_sample
            })

        f()

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):

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

        return rval

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

    def __init__(self, n_classes, layer_name, irange = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None,
                 copies = 1):
        """
            copies: we regard the layer as being copied <copies> times
                   all sample and mean field states are the *average* of
                   all of these copies, and the weights to each copy are
                   tied.
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self

        assert isinstance(n_classes, int)

        self.output_space = VectorSpace(n_classes)
        self.b = sharedX( np.zeros((n_classes,)), name = 'softmax_b')

    def get_lr_scalers(self):

        rval = {}

        # Patch old pickle files
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_total_state_space(self):
        return self.output_space

    def get_monitoring_channels_from_state(self, state):

        mx = state.max(axis=1)

        return {
                'mean_max_class' : mx.mean(),
                'max_max_class' : mx.max(),
                'min_max_class' : mx.min()
        }

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
                    W[idx, i] = rng.randn()

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

    def set_biases(self, biases):
        self.b.set_value(biases)

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
        log_prob = z - T.exp(z).sum(axis=1).dimshuffle(0, 'x')
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

        p_state = sharedX( self.output_space.get_origin_batch(
            num_examples))


        t2 = time.time()

        f = function([], updates = {
            h_state : h_sample
            })

        t3 = time.time()

        f()

        t4 = time.time()

        print str(self)+'.make_state took',t4-t1
        print '\tcompose time:',t2-t1
        print '\tcompile time:',t3-t2
        print '\texecute time:',t4-t3

        h_state.name = 'softmax_sample_shared'

        return h_state

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def expected_energy_term(self, state, average, state_below, average_below):

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
            for i in xrange(niter-1):
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
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat

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
