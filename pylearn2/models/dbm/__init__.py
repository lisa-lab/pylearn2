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

import theano
from theano.compat.python2x import OrderedDict
from theano import gof
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.base import Block
from pylearn2.utils import block_gradient
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip

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


class InferenceProcedure(object):
    """
    .. todo::

        WRITEME
    """
    def set_dbm(self, dbm):
        """
        .. todo::

            WRITEME
        """
        self.dbm = dbm

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement mf.")

    def set_batch_size(self, batch_size):
        """
        If the inference procedure is dependent on a batch size at all, makes
        the necessary internal configurations to work with that batch size.
        """
        # TODO : was this supposed to be implemented?


class WeightDoubling(InferenceProcedure):
    """
    .. todo::

        WRITEME
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """

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


class SuperWeightDoubling(WeightDoubling):
    """
    .. todo::

        WRITEME
    """
    def multi_infer(self, V, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """

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
        .. todo::

            WRITEME properly

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V.
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat

        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in `model.output_space`, i.e. in the output space of \
            the last hidden layer. (It's not really a hidden layer anymore, \
            but oh well. It's convenient to code it this way because the \
            labels are sort of "on top" of everything else.) *** Y is always \
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            Theano batch in `model.input_space`. Should be all binary, with \
            1s indicating that the corresponding element of X should be \
            "dropped", i.e. hidden from the algorithm and filled in as part \
            of the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is \
            a single categorical variable. `drop_mask_Y` is a binary mask \
            specifying which *rows* to drop.
        return_history : bool, optional
            WRITEME
        noise : bool, optional
            WRITEME
        niter : int, optional
            WRITEME
        block_grad : WRITEME

        Returns
        -------
        WRITEME
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
        """
        .. todo::

            WRITEME
        """

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
    """
    .. todo::

        WRITEME
    """

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME properly

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat

        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in `model.output_space`, i.e. in the output space of \
            the last hidden layer. (It's not really a hidden layer anymore, \
            but oh well. It's convenient to code it this way because the \
            labels are sort of "on top" of everything else.) *** Y is always \
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            Theano batch in `model.input_space`. Should be all binary, with \
            1s indicating that the corresponding element of X should be \
            "dropped", i.e. hidden from the algorithm and filled in as part \
            of the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is \
            a single categorical variable. `drop_mask_Y` is a binary mask \
            specifying which *rows* to drop.
        return_history : bool, optional
            WRITEME
        noise : bool, optional
            WRITEME
        niter : int, optional
            WRITEME
        block_grad : WRITEME

        Returns
        -------
        WRITEME
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
        """
        .. todo::

            WRITEME
        """

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
        .. todo::

            WRITEME properly

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V.
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in model.output_space, ie, in the output space of \
            the last hidden layer (it's not really a hidden layer anymore, \
            but oh well. It's convenient to code it this way because the \
            labels are sort of "on top" of everything else). *** Y is always \
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            A theano batch in `model.input_space`. Should be all binary, with \
            1s indicating that the corresponding element of X should be \
            "dropped", ie, hidden from the algorithm and filled in as part of \
            the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is \
            a single categorical variable. `drop_mask_Y` is a binary mask \
            specifying which *rows* to drop.
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
        """
        .. todo::

            WRITEME
        """

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
        .. todo::

            WRITEME properly

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V.
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in model.output_space, ie, in the output space of \
            the last hidden layer (it's not really a hidden layer anymore, \
            but oh well. It's convenient to code it this way because the \
            labels are sort of "on top" of everything else). *** Y is always \
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            A theano batch in `model.input_space`. Should be all binary, with \
            1s indicating that the corresponding element of X should be \
            "dropped", ie, hidden from the algorithm and filled in as part of \
            the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is \
            a single categorical variable. `drop_mask_Y` is a binary mask \
            specifying which *rows* to drop.
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

# Make known modules inside this package
# this needs to come after e.g. flatten(), since DBM depends on flatten
from pylearn2.models.dbm.dbm import DBM
from pylearn2.models.dbm.layer import Layer, VisibleLayer, HiddenLayer, BinaryVectorMaxPool, Softmax