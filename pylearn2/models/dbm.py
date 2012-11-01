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
import time
import warnings

from theano import function
from theano.gof.op import get_debug_values
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_zip
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

class HiddenLayer(Layer):
    """
    Abstract class.
    A layer of a DBM that may be used as a hidden layer.
    """

    def downward_state(self, total_state):
        return total_state

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
            include_prob = 1.0,
            init_bias = 0.):
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
            for i in xrange(self.detector_layer_dim):
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0:
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

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
        assert isinstance(coeff, float)
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

            for key, val in [
                    ('max_max', v_max.max()),
                    ('max_mean', v_max.mean()),
                    ('max_min', v_max.min()),
                    ('min_max', v_min.max()),
                    ('min_mean', v_min.mean()),
                    ('min_max', v_min.max()),
                    ('range_max', v_range.max()),
                    ('range_mean', v_range.mean()),
                    ('range_min', v_range.min()),
                    ('mean_max', v_mean.max()),
                    ('mean_mean', v_mean.mean()),
                    ('mean_min', v_mean.min())
                    ]:
                rval[prefix+key] = val

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
            assert isinstance(target, float)
            assert isinstance(coeff, float)
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
            assert all([isinstance(elem, float) for elem in [t, c, e]])
            if c == 0.:
                continue
            m = s.mean(axis=0)
            assert m.ndim == 1
            rval += T.maximum(abs(m-t)-e,0.).mean()*c

        return rval

    def get_sampling_updates(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):

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

    def make_state(self, num_examples, numpy_rng):
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(num_examples)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.b

        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 16))

        p_exp, h_exp, p_sample, h_sample = max_pool_channels(
                z = default_z,
                pool_size = self.pool_size,
                theano_rng = theano_rng)

        p_state = sharedX( self.output_space.get_origin_batch(
            num_examples))

        t2 = time.time()

        f = function([], updates = {
            p_state : p_sample,
            h_state : h_sample
            })

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
