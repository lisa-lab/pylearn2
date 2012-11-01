"""
This module contains functionality related to Deep Boltzmann Machines.
They are implemented generically in order to make it easy to support
convolution versions, etc.

Most of the code needed to actually use the DBM is still in Ian's private
repo. He'll move it here after he has a paper, or if you need it, you
can write to him to make an arrangement.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np
import sys
import warnings

from theano.printing import Print
import theano.tensor as T

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
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
            niter):
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


class DBM_Layer(Model):
    """
    A layer of a DBM.
    May only belong to one DBM.

    Each layer has a state ("total state") that can be split into
    the piece that is visible to the layer above ("upward state")
    and the piece that is visible to the layer below ("downward state").

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
        assert self.get_debm() is None
        self.dbm = dbm

    def get_total_state_space(self, state):
        """
        Returns the Space that the layer's total state lives in.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +\
                "get_total_state_space()")

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

        raise NotImplementedError("%s doesn't implement get_sampling_updates" %
                type(self))

class DBM_VisibleLayer(DBM_Layer):
    """
    A layer of a DBM that may be used as a visible layer.
    """

class BinaryVisLayer(DBM_VisibleLayer):
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

    def get_total_state_space(self):
        return self.get_input_space()

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

