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
import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.base import Block
from pylearn2.utils import block_gradient
from pylearn2.utils import py_integer_types

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


def init_sigmoid_bias_from_marginals(dataset, use_y = False):
    """
    Returns b such that sigmoid(b) has the same marginals as the
    data. Assumes dataset contains a design matrix. If use_y is
    true, sigmoid(b) will have the same marginals as the targets,
    rather than the features.

    Parameters
    ----------
    dataset : WRITEME
    use_y : WRITEME
    """
    if use_y:
        X = dataset.y
    else:
        X = dataset.get_design_matrix()
    return init_sigmoid_bias_from_array(X)

def init_sigmoid_bias_from_array(arr):
    """
    .. todo::

        WRITEME
    """
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


class SamplingProcedure(object):
    """
    Procedure for sampling from a DBM.
    """
    def set_dbm(self, dbm):
        """
        .. todo::

            WRITEME
        """
        self.dbm = dbm

    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None,
               num_steps=1):
        """
        Samples from self.dbm using `layer_to_state` as starting values.

        Parameters
        ----------
        layer_to_state : dict
            Maps the DBM's Layer instances to theano variables representing \
            batches of samples of them.
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            WRITEME
        layer_to_clamp : dict, optional
            Maps Layers to bools. If a layer is not in the dictionary, \
            defaults to False. True indicates that this layer should be \
            clamped, so we are sampling from a conditional distribution \
            rather than the joint distribution.

        Returns
        -------
        layer_to_updated_state : dict
            Maps the DBM's Layer instances to theano variables representing \
            batches of updated samples of them.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "sample.")


class GibbsEvenOdd(SamplingProcedure):
    """
    The specific sampling schedule used to sample all of the even-idexed
    layers of model.hidden_layers, then the visible layer and all the
    odd-indexed layers.
    """
    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None,
               num_steps=1):
        """
        .. todo::

            WRITEME
        """
        # Validate num_steps
        assert isinstance(num_steps, py_integer_types)
        assert num_steps > 0

        # Implement the num_steps > 1 case by repeatedly calling the
        # num_steps == 1 case
        if num_steps != 1:
            for i in xrange(num_steps):
                layer_to_state = self.sample(layer_to_state, theano_rng,
                                             layer_to_clamp, num_steps=1)
            return layer_to_state

        # The rest of the function is the num_steps = 1 case
        # Current code assumes this, though we could certainly relax this
        # constraint
        assert len(self.dbm.hidden_layers) > 0

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully
        # populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert (key is self.dbm.visible_layer or
                    key in self.dbm.hidden_layers)

        for layer in [self.dbm.visible_layer] + self.dbm.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        # Assemble the return value
        layer_to_updated = OrderedDict()

        for i, this_layer in list(enumerate(self.dbm.hidden_layers))[::2]:
            # Iteration i does the Gibbs step for hidden_layers[i]

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            if i == 0:
                layer_below = self.dbm.visible_layer
            else:
                layer_below = self.dbm.hidden_layers[i-1]
            state_below = layer_to_state[layer_below]
            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.dbm.hidden_layers):
                layer_above = self.dbm.hidden_layers[i + 1]
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
                this_sample = this_layer.sample(state_below=state_below,
                                                state_above=state_above,
                                                layer_above=layer_above,
                                                theano_rng=theano_rng)

            layer_to_updated[this_layer] = this_sample

        #Sample the visible layer
        vis_state = layer_to_state[self.dbm.visible_layer]
        if layer_to_clamp[self.dbm.visible_layer]:
            vis_sample = vis_state
        else:
            first_hid = self.dbm.hidden_layers[0]
            state_above = layer_to_updated[first_hid]
            state_above = first_hid.downward_state(state_above)

            vis_sample = self.dbm.visible_layer.sample(state_above=state_above,
                                                       layer_above=first_hid,
                                                       theano_rng=theano_rng)
        layer_to_updated[self.dbm.visible_layer] = vis_sample

        # Sample the odd-numbered layers
        for i, this_layer in list(enumerate(self.dbm.hidden_layers))[1::2]:

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            layer_below = self.dbm.hidden_layers[i-1]

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
            if i + 1 < len(self.dbm.hidden_layers):
                layer_above = self.dbm.hidden_layers[i + 1]
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
                this_sample = this_layer.sample(state_below=state_below,
                                                state_above=state_above,
                                                layer_above=layer_above,
                                                theano_rng=theano_rng)

            layer_to_updated[this_layer] = this_sample

        # Check that all layers were updated
        assert all([layer in layer_to_updated for layer in layer_to_state])
        # Check that we didn't accidentally treat any other object as a layer
        assert all([layer in layer_to_state for layer in layer_to_updated])
        # Check that clamping worked
        assert all([(layer_to_state[layer] is layer_to_updated[layer]) ==
                    layer_to_clamp[layer] for layer in layer_to_state])

        return layer_to_updated


class DBMSampler(Block):
    """
    A Block used to sample from the last layer of a DBM with one hidden layer.
    """
    def __init__(self, dbm):
        """
        .. todo::

            WRITEME
        """
        super(DBMSampler, self).__init__()
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        self.dbm = dbm
        assert len(self.dbm.hidden_layers) == 1

    def __call__(self, inputs):
        """
        .. todo::

            WRITEME
        """
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

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.dbm.get_input_space()

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.dbm.get_output_space()


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

    Parameters
    ----------
    l : WRITEME

    Returns
    -------
    WRITEME
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
    """
    .. todo::

        WRITEME
    """
    new = []
    for elem in l:
        if isinstance(elem, (list, tuple)):
            new.append(block(elem))
        else:
            new.append(block_gradient(elem))
    if isinstance(l, tuple):
        return tuple(new)
    return new


# Make known modules inside this package
# this needs to come after e.g. flatten(), since DBM depends on flatten
from pylearn2.models.dbm.layer import Layer, VisibleLayer, HiddenLayer, BinaryVectorMaxPool, Softmax
from pylearn2.models.dbm.inference_procedure import InferenceProcedure, WeightDoubling, SuperWeightDoubling
from pylearn2.models.dbm.dbm import DBM