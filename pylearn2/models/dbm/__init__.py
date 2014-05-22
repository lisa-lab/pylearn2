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
__maintainer__ = "LISA Lab"

import logging
import numpy as np
import sys

from theano.compat.python2x import OrderedDict

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.blocks import Block
from pylearn2.utils import block_gradient
from pylearn2.utils.rng import make_theano_rng


logger = logging.getLogger(__name__)

logger.debug("DBM changing the recursion limit.")
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


class DBMSampler(Block):
    """
    A Block used to sample from the last layer of a DBM with one hidden layer.

    Parameters
    ----------
    dbm : WRITEME
    """
    def __init__(self, dbm):
        super(DBMSampler, self).__init__()
        self.theano_rng = make_theano_rng(None, 2012+10+14, which_method="binomial")
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
# this needs to come after e.g. flatten(), since DBM depends on flatten()
from pylearn2.models.dbm.dbm import DBM
from pylearn2.models.dbm.inference_procedure import BiasInit
from pylearn2.models.dbm.inference_procedure import InferenceProcedure
from pylearn2.models.dbm.inference_procedure import MoreConsistent
from pylearn2.models.dbm.inference_procedure import MoreConsistent2
from pylearn2.models.dbm.inference_procedure import SuperWeightDoubling
from pylearn2.models.dbm.inference_procedure import WeightDoubling
from pylearn2.models.dbm.layer import BinaryVector
from pylearn2.models.dbm.layer import BinaryVectorMaxPool
from pylearn2.models.dbm.layer import BVMP_Gaussian
from pylearn2.models.dbm.layer import CompositeLayer
from pylearn2.models.dbm.layer import ConvMaxPool
from pylearn2.models.dbm.layer import ConvC01B_MaxPool
from pylearn2.models.dbm.layer import GaussianVisLayer
from pylearn2.models.dbm.layer import HiddenLayer
from pylearn2.models.dbm.layer import Layer
from pylearn2.models.dbm.layer import VisibleLayer
from pylearn2.models.dbm.layer import Softmax
from pylearn2.models.dbm.sampling_procedure import SamplingProcedure
