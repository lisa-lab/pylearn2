"""
Common DBM Layer classes
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import functools
import logging
import numpy as np
import time
import warnings

from theano import tensor as T, function, config
import theano
from theano.compat import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print

from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels, max_pool_b01c, max_pool, max_pool_c01b
from pylearn2.linear.conv2d import make_random_conv2D, make_sparse_random_conv2D
from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models import Model
from pylearn2.models.dbm import init_sigmoid_bias_from_marginals
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace, Space
from pylearn2.utils import is_block_gradient
from pylearn2.utils import sharedX, safe_zip, py_integer_types, block_gradient
from pylearn2.utils.rng import make_theano_rng
"""
.. todo::

    WRITEME
"""
from pylearn2.utils import safe_union


logger = logging.getLogger(__name__)


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

        Parameters
        ----------
        dbm : WRITEME
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
        .. todo::

            WRITEME
        """
        return OrderedDict()

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """
        return OrderedDict()

    def upward_state(self, total_state):
        """
        Takes total_state and turns it into the state that layer_above should
        see when computing P( layer_above | this_layer).

        So far this has two uses:

        * If this layer consists of a detector sub-layer h that is pooled
          into a pooling layer p, then total_state = (p,h) but layer_above
          should only see p.
        * If the conditional P( layer_above | this_layer) depends on
          parameters of this_layer, sometimes you can play games with
          the state to avoid needing the layers to communicate. So far
          the only instance of this usage is when the visible layer
          is N( Wh, beta). This makes the hidden layer be
          sigmoid( v beta W + b). Rather than having the hidden layer
          explicitly know about beta, we can just pass v beta as
          the upward state.

        Parameters
        ----------
        total_state : WRITEME

        Notes
        -----
        This method should work both for computing sampling updates
        and for computing mean field updates. So far I haven't encountered
        a case where it needs to do different things for those two
        contexts.
        """
        return total_state

    def make_state(self, num_examples, numpy_rng):
        """
        Returns a shared variable containing an actual state (not a mean field
        state) for this variable.

        Parameters
        ----------
        num_examples : WRITEME
        numpy_rng : WRITEME

        Returns
        -------
        WRITEME
        """

        raise NotImplementedError("%s doesn't implement make_state" %
                type(self))

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        Returns a theano symbolic variable containing an actual state (not a
        mean field state) for this variable.

        Parameters
        ----------
        num_examples : WRITEME
        numpy_rng : WRITEME

        Returns
        -------
        WRITEME
        """

        raise NotImplementedError("%s doesn't implement make_symbolic_state" %
                                  type(self))

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
        Returns an expression for samples of this layer's state, conditioned on
        the layers above and below Should be valid as an update to the shared
        variable returned by self.make_state

        Parameters
        ----------
        state_below : WRITEME
            Corresponds to layer_below.upward_state(full_state_below),
            where full_state_below is the same kind of object as you get
            out of layer_below.make_state
        state_above : WRITEME
            Corresponds to layer_above.downward_state(full_state_above)

        theano_rng : WRITEME
            An MRG_RandomStreams instance

        Returns
        -------
        WRITEME

        Notes
        -----
        This can return multiple expressions if this layer's total state
        consists of more than one shared variable.
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

        - involve this layer only
        - if there is a layer below, include terms that involve both this layer
          and the layer below

        Do not include terms that involve the layer below only.
        Do not include any terms that involve the layer above, if it
        exists, in any way (the interface doesn't let you see the layer
        above anyway).

        Parameters
        ----------
        state_below : WRITEME
            Upward state of the layer below.
        state : WRITEME
            Total state of this layer
        average_below : bool
            If True, the layer below is one of the variables to integrate
            over in the expectation, and state_below gives its variational
            parameters. If False, that layer is to be held constant and
            state_below gives a set of assignments to it average: like
            average_below, but for 'state' rather than 'state_below'

        Returns
        -------
        rval : tensor_like
            A 1D theano tensor giving the expected energy term for each example
        """
        raise NotImplementedError(str(type(self))+" does not implement expected_energy_term.")

    def finalize_initialization(self):
        """
        Some layers' initialization depends on layer above being initialized,
        which is why this method is called after `set_input_space` has been
        called.
        """
        pass


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
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()


class HiddenLayer(Layer):
    """
    Abstract class.
    A layer of a DBM that may be used as a hidden layer.
    """

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def get_stdev_rewards(self, state, coeffs):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement get_stdev_rewards")

    def get_range_rewards(self, state, coeffs):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement get_range_rewards")

    def get_l1_act_cost(self, state, target, coeff, eps):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement get_l1_act_cost")

    def get_l2_act_cost(self, state, target, coeff):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement get_l2_act_cost")


class BinaryVector(VisibleLayer):
    """
    A DBM visible layer consisting of binary random variables living
    in a VectorSpace.

    Parameters
    ----------
    nvis : int
        Dimension of the space
    bias_from_marginals : pylearn2.datasets.dataset.Dataset
        Dataset, whose marginals are used to initialize the visible biases
    center : bool
        WRITEME
    copies : int
        WRITEME
    """
    def __init__(self,
            nvis,
            bias_from_marginals = None,
            center = False,
            copies = 1, learn_init_inpainting_state = False):

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
        """
        Returns
        -------
        biases : ndarray
            The numpy value of the biases
        """
        return self.bias.get_value()

    def set_biases(self, biases, recenter=False):
        """
        .. todo::

            WRITEME
        """
        self.bias.set_value(biases)
        if recenter:
            assert self.center
            self.offset.set_value(sigmoid_numpy(self.bias.get_value()))

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        return [self.bias]

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
        .. todo::

            WRITEME
        """


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
        """
        .. todo::

            WRITEME
        """
        msg = layer_above.downward_message(state_above)
        mu = self.bias

        z = msg + mu

        rval = T.nnet.sigmoid(z)

        return rval


    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'copies'):
            self.copies = 1
        if self.copies != 1:
            raise NotImplementedError()
        mean = T.nnet.sigmoid(self.bias)
        rval = theano_rng.binomial(size=(num_examples, self.nvis), p=mean,
                                   dtype=theano.config.floatX)

        return rval

    def expected_energy_term(self, state, average, state_below = None, average_below = None):
        """
        .. todo::

            WRITEME
        """

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

    def init_inpainting_state(self, V, drop_mask, noise = False, return_unmasked = False):
        """
        .. todo::

            WRITEME
        """
        assert drop_mask is None or drop_mask.ndim > 1

        unmasked = T.nnet.sigmoid(self.bias.dimshuffle('x',0))
        # this condition is needed later if unmasked is used as V_hat
        assert unmasked.ndim == 2
        # this condition is also needed later if unmasked is used as V_hat
        assert hasattr(unmasked.owner.op, 'scalar_op')
        if drop_mask is not None:
            masked_mean = unmasked * drop_mask
        else:
            masked_mean = unmasked
        if not hasattr(self, 'learn_init_inpainting_state'):
            self.learn_init_inpainting_state = 0
        if not self.learn_init_inpainting_state:
            masked_mean = block_gradient(masked_mean)
        masked_mean.name = 'masked_mean'

        if noise:
            theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
            # we want a set of random mean field parameters, not binary samples
            unmasked = T.nnet.sigmoid(theano_rng.normal(avg = 0.,
                    std = 1., size = masked_mean.shape,
                    dtype = masked_mean.dtype))
            masked_mean = unmasked * drop_mask
            masked_mean.name = 'masked_noise'

        if drop_mask is None:
            rval = masked_mean
        else:
            masked_V  = V  * (1-drop_mask)
            rval = masked_mean + masked_V
        rval.name = 'init_inpainting_state'

        if return_unmasked:
            assert unmasked.ndim > 1
            return rval, unmasked

        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask = None, V = None, return_unmasked = False):
        """
        .. todo::

            WRITEME
        """
        msg = layer_above.downward_message(state_above)
        mu = self.bias

        z = msg + mu
        z.name = 'inpainting_z_[unknown_iter]'

        unmasked = T.nnet.sigmoid(z)

        if drop_mask is not None:
            rval = drop_mask * unmasked + (1-drop_mask) * V
        else:
            rval = unmasked

        rval.name = 'inpainted_V[unknown_iter]'

        if return_unmasked:
            owner = unmasked.owner
            assert owner is not None
            op = owner.op
            assert hasattr(op, 'scalar_op')
            assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
            return rval, unmasked

        return rval


    def recons_cost(self, V, V_hat_unmasked, drop_mask = None, use_sum=False):
        """
        .. todo::

            WRITEME
        """
        if use_sum:
            raise NotImplementedError()

        V_hat = V_hat_unmasked

        assert hasattr(V_hat, 'owner')
        owner = V_hat.owner
        assert owner is not None
        op = owner.op
        block_grad = False
        if is_block_gradient(op):
            assert isinstance(op.scalar_op, theano.scalar.Identity)
            block_grad = True
            real, = owner.inputs
            owner = real.owner
            op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected V_hat_unmasked to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z ,= owner.inputs
        if block_grad:
            z = block_gradient(z)

        if V.ndim != V_hat.ndim:
            raise ValueError("V and V_hat_unmasked should have same ndim, but are %d and %d." % (V.ndim, V_hat.ndim))
        unmasked_cost = V * T.nnet.softplus(-z) + (1 - V) * T.nnet.softplus(z)
        assert unmasked_cost.ndim == V_hat.ndim

        if drop_mask is None:
            masked_cost = unmasked_cost
        else:
            masked_cost = drop_mask * unmasked_cost

        return masked_cost.mean()

class BinaryVectorMaxPool(HiddenLayer):
    """
    A hidden layer that does max-pooling on binary vectors.
    It has two sublayers, the detector layer and the pooling
    layer. The detector layer is its downward state and the pooling
    layer is its upward state.

    Parameters
    ----------
    detector_layer_dim : WRITEME
    pool_size : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    sparse_init : WRITEME
    sparse_stdev : WRITEME
    include_prob : , optional
        Probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is
        initialized to 0.
    init_bias : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    center : WRITEME
    mask_weights : WRITEME
    max_col_norm : WRITEME
    copies : WRITEME
    """
    # TODO: this layer uses (pooled, detector) as its total state,
    #       which can be confusing when listing all the states in
    #       the network left to right. Change this and
    #       pylearn2.expr.probabilistic_max_pooling to use
    #       (detector, pooled)

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
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if self.center:
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset = sharedX(sigmoid_numpy(self.b.get_value()))

    def get_lr_scalers(self):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME

        Notes
        -----
        This resets parameters!
        """

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

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):

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
        """
        .. todo::

            WRITEME
        """
        return CompositeSpace((self.output_space, self.h_space))

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        return W.get_value()

    def set_weights(self, weights):
        """
        .. todo::

            WRITEME
        """
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter = False):
        """
        .. todo::

            WRITEME
        """
        self.b.set_value(biases)
        if recenter:
            assert self.center
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset.set_value(sigmoid_numpy(self.b.get_value()))

    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        return self.b.get_value()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ('v', 'h')

    def get_weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W ,= self.transformer.get_params()

        W = W.T

        W = W.reshape((self.detector_layer_dim, self.input_space.shape[0],
            self.input_space.shape[1], self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            return h - self.offset

        return h * self.copies

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        self.h_space.validate(downward_state)
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval * self.copies

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size, self.detector_layer_dim).astype(self.b.dtype) + \
                self.b.dimshuffle('x', 0)
        rval = max_pool_channels(z = z,
                pool_size = self.pool_size)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
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

        theano_rng = make_theano_rng(None, numpy_rng.randint(2 ** 16), which_method="binomial")

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
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """

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
        Used to implement TorontoSparsity. Unclear exactly what properties of
        it are important or how to implement it for other layers.

        Properties it must have: output is same kind of data structure (ie,
        tuple of theano 2-tensors) as mf_update.

        Properties it probably should have for other layer types: an
        infinitesimal change in state_below or the parameters should cause the
        same sign of change in the output of linear_feed_forward_approximation
        and in mf_update

        Should not have any non-linearities that cause the gradient to shrink

        Should disregard top-down feedback

        Parameters
        ----------
        state_below : WRITEME
        """

        z = self.transformer.lmul(state_below) + self.b

        if self.pool_size != 1:
            # Should probably implement sum pooling for the non-pooled version,
            # but in reality it's not totally clear what the right answer is
            raise NotImplementedError()

        return z, z

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):
        """
        .. todo::

            WRITEME
        """

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
    """
    .. todo::

        WRITEME
    """

    presynaptic_name = "presynaptic_Y_hat"

    def __init__(self, n_classes, layer_name, irange = None,
                 sparse_init = None, sparse_istdev = 1., W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 copies = 1, center = False,
                 learn_init_inpainting_state = True):
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

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):

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
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        return self.output_space

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """

        mx = state.max(axis=1)

        return OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        """
        .. todo::

            WRITEME
        """
        self.W.set_value(weights)

    def set_biases(self, biases, recenter=False):
        """
        .. todo::

            WRITEME
        """
        self.b.set_value(biases)
        if recenter:
            assert self.center
            self.offset.set_value( (np.exp(biases) / np.exp(biases).sum()).astype(self.offset.dtype))

    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        return self.b.get_value()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ('v', 'h')

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
        .. todo::

            WRITEME
        """


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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """

        if not hasattr(self, 'copies'):
            self.copies = 1

        rval =  T.dot(downward_state, self.W.T) * self.copies

        rval = self.desired_space.format_as(rval, self.input_space)

        return rval

    def recons_cost(self, Y, Y_hat_unmasked, drop_mask_Y, scale):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        rval =  T.nnet.softmax(self.b.dimshuffle('x', 0)) + T.alloc(0., self.dbm.batch_size, self.n_classes).astype(config.floatX)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        if self.copies != 1:
            raise NotImplementedError("need to make self.copies samples and average them together.")

        t1 = time.time()

        empty_input = self.output_space.get_origin_batch(num_examples)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.b

        theano_rng = make_theano_rng(None, numpy_rng.randint(2 ** 16),
                                     which_method="binomial")

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

        logger.info('{0}.make_state took {1}'.format(self, t4-t1))
        logger.info('\tcompose time: {0}'.format(t2-t1))
        logger.info('\tcompile time: {0}'.format(t3-t2))
        logger.info('\texecute time: {0}'.format(t4-t3))

        h_state.name = 'softmax_sample_shared'

        return h_state

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def upward_state(self, state):
        """
        .. todo::

            WRITEME
        """
        if self.center:
            return state - self.offset
        return state

    def downward_state(self, state):
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'center'):
            self.center = False
        if self.center:
            """TODO: write a unit test verifying that inference or sampling
                     below a centered Softmax layer works"""
            return state - self.offset
        return state

    def expected_energy_term(self, state, average, state_below, average_below):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        if noise:
            theano_rng = make_theano_rng(None, 2012+10+30, which_method="binomial")
            return T.nnet.softmax(theano_rng.normal(avg=0., size=Y.shape, std=1., dtype='float32'))
        rval =  T.nnet.softmax(self.b)
        if not hasattr(self, 'learn_init_inpainting_state'):
            self.learn_init_inpainting_state = 1
        if not self.learn_init_inpainting_state:
            rval = block_gradient(rval)
        return rval

    def install_presynaptic_outputs(self, outputs_dict, batch_size):
        """
        .. todo::

            WRITEME
        """

        assert self.presynaptic_name not in outputs_dict
        outputs_dict[self.presynaptic_name] = self.output_space.make_shared_batch(batch_size, self.presynaptic_name)


class GaussianVisLayer(VisibleLayer):
    """
    Implements a visible layer that is conditionally gaussian with
    diagonal variance. The layer lives in a Conv2DSpace.

    Parameters
    ----------
    rows, cols, channels : WRITEME
        the shape of the space
    learn_init_inpainting : bool, optional
        WRITEME
    nvis : WRITEME
    init_beta : WRITEME
        the initial value of the precision parameter
    min_beta : WRITEME
        clip beta so it is at least this big (default 1)
    init_mu : WRITEME
        the initial value of the mean parameter
    tie_beta : WRITEME
        None or a string specifying how to tie beta 'locations' = tie beta
        across locations, ie beta should be a vector with one elem per channel
    tie_mu : WRITEME
        None or a string specifying how to tie mu 'locations' = tie mu across
        locations, ie mu should be a vector with one elem per channel
    bias_from_marginals : WRITEME
    beta_lr_scale : WRITEME
    axes : tuple
        WRITEME
    """
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
            # To make GaussianVisLayer compatible with any axis ordering
            self.batch_axis=list(axes).index('b')
            self.axes_to_sum = range(len(axes))
            self.axes_to_sum.remove(self.batch_axis)
        else:
            assert rows is None
            assert cols is None
            assert channels is None
            self.space = VectorSpace(nvis)
            self.axes_to_sum = 1
            self.batch_axis = None
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
        """
        .. todo::

            WRITEME
        """
        rval = OrderedDict()

        rval['beta_min'] = self.beta.min()
        rval['beta_mean'] = self.beta.mean()
        rval['beta_max'] = self.beta.max()

        return rval


    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        if self.mu is None:
            return [self.beta]
        return [self.beta, self.mu]

    def get_lr_scalers(self):
        """
        .. todo::

            WRITEME
        """
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
            logger.warning("mu lr_scaler hardcoded to 1/sharing")

        return rval

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        if self.beta in updates:
            updated_beta = updates[self.beta]
            updates[self.beta] = T.clip(updated_beta,
                    self.min_beta,1e6)

    def set_biases(self, bias):
        """
        Set mean parameter

        Parameters
        ----------
        bias: WRITEME
            Vector of size nvis
        """
        self.mu = sharedX(bias, name = 'mu')

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
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """

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
            theano_rng = make_theano_rng(None, 42, which_method="binomial")
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
        """
        .. todo::

            WRITEME
        """
        assert state_below is None
        assert average_below is None
        self.space.validate(state)
        if average:
            raise NotImplementedError(str(type(self))+" doesn't support integrating out variational parameters yet.")
        else:
            rval =  0.5 * (self.beta * T.sqr(state - self.mu)).sum(axis=self.axes_to_sum)
        assert rval.ndim == 1
        return rval


    def inpaint_update(self, state_above, layer_above, drop_mask = None, V = None,
                        return_unmasked = False):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """

        assert state_below is None
        msg = layer_above.downward_message(state_above)
        mu = self.mu

        z = msg + mu
        rval = theano_rng.normal(size = z.shape, avg = z, dtype = z.dtype,
                       std = 1. / T.sqrt(self.beta))
        return rval

    def recons_cost(self, V, V_hat_unmasked, drop_mask = None, use_sum=False):
        """
        .. todo::

            WRITEME
        """

        return self._recons_cost(V=V, V_hat_unmasked=V_hat_unmasked, drop_mask=drop_mask, use_sum=use_sum, beta=self.beta)


    def _recons_cost(self, V, V_hat_unmasked, beta, drop_mask=None, use_sum=False):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if self.nvis is None and total_state.ndim != 4:
            raise ValueError("total_state should have 4 dimensions, has "+str(total_state.ndim))
        assert total_state is not None
        V = total_state
        self.input_space.validate(V)
        upward_state = (V - self.broadcasted_mu()) * self.broadcasted_beta()
        return upward_state

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """

        outputs_dict['output_V_weighted_pred_sum'] = self.space.make_shared_batch(batch_size)

    def ensemble_prediction(self, symbolic, outputs_dict, ensemble):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """

        beta = sum(ensemble.get_ensemble_variants(self.beta)) / ensemble.num_copies

        return self._recons_cost(V=V, V_hat_unmasked=V_hat_unmasked, beta=beta, drop_mask=drop_mask,
            use_sum=use_sum)


class ConvMaxPool(HiddenLayer):
    """
    .. todo::

        WRITEME
    """

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
        self.__dict__.update(locals())
        del self.self

        assert (irange is None) != (sparse_init is None)

        self.b = sharedX( np.zeros((output_channels,)) + init_bias, name = layer_name + '_b')
        assert border_mode in ['full','valid']

    def broadcasted_bias(self):
        """
        .. todo::

            WRITEME
        """

        assert self.b.ndim == 1

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)


    def get_total_state_space(self):
        """
        .. todo::

            WRITEME
        """
        return CompositeSpace((self.h_space, self.output_space))

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
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

        logger.info('{0}: detector shape: {1} '
                    'pool shape: {2}'.format(self.layer_name,
                                             self.h_space.shape,
                                             self.output_space.shape))

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
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None

        return [ W, self.b]

    def state_to_b01c(self, state):
        """
        .. todo::

            WRITEME
        """

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_range_rewards(self, state, coeffs):
        """
        .. todo::

            WRITEME
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
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if self.scale_by_sharing:
            # scale each learning rate by 1 / # times param is reused
            h_rows, h_cols = self.h_space.shape
            num_h = float(h_rows * h_cols)
            return OrderedDict([(self.transformer._filters, 1./num_h),
                     (self.b, 1. / num_h)])
        else:
            return OrderedDict()

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return p

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return h

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        W , = self.transformer.get_params()
        return coeffs * T.sqr(W).sum()



    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        self.h_space.validate(downward_state)
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw,(outp,rows,cols,inp))


    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        """ Returns a shared variable containing an actual state
           (not a mean field state) for this variable.
        """

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(self.dbm.batch_size)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.broadcasted_bias()

        theano_rng = make_theano_rng(None, numpy_rng.randint(2 ** 16),
                                     which_method="binomial")

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

        logger.info('{0}.make_state took'.format(self, t4-t1))
        logger.info('\tcompose time: {0}'.format(t2-t1))
        logger.info('\tcompile time: {0}'.format(t3-t2))
        logger.info('\texecute time: {0}'.format(t4-t3))

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):
        """
        .. todo::

            WRITEME
        """

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
    """
    .. todo::

        WRITEME
    """

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
        self.__dict__.update(locals())
        del self.self

        assert (irange is None) != (sparse_init is None)
        self.output_axes = ('c', 0, 1, 'b')
        self.detector_channels = output_channels
        self.tied_b = 1

    def broadcasted_bias(self):
        """
        .. todo::

            WRITEME
        """

        if self.b.ndim != 1:
            raise NotImplementedError()

        shuffle = [ 'x' ] * 4
        shuffle[self.output_axes.index('c')] = 0

        return self.b.dimshuffle(*shuffle)


    def get_total_state_space(self):
        """
        .. todo::

            WRITEME
        """
        return CompositeSpace((self.h_space, self.output_space))

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
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

        logger.info('{0} : detector shape: {1} '
                    'pool shape: {2}'.format(self.layer_name,
                                             self.h_space.shape,
                                             self.output_space.shape))

        assert tuple(self.output_axes) == ('c', 0, 1, 'b')
        self.max_pool = max_pool_c01b

        if self.center:
            p_ofs, h_ofs = self.init_mf_state()
            self.p_offset = sharedX(self.output_space.get_origin(), 'p_offset')
            self.h_offset = sharedX(self.h_space.get_origin(), 'h_offset')
            f = function([], updates={self.p_offset: p_ofs[:,:,:,0], self.h_offset: h_ofs[:,:,:,0]})
            f()


    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None

        return [ W, self.b]

    def state_to_b01c(self, state):
        """
        .. todo::

            WRITEME
        """

        if tuple(self.output_axes) == ('b',0,1,'c'):
            return state
        return [ Conv2DSpace.convert(elem, self.output_axes, ('b', 0, 1, 'c'))
                for elem in state ]

    def get_range_rewards(self, state, coeffs):
        """
        .. todo::

            WRITEME
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
        .. todo::

            WRITEME properly

        Parameters
        ----------
        state : WRITEME
        target : WRITEME
            if pools contain more than one element, should be a list
            with two elements. the first element is for the pooling
            units and the second for the detector units.
        coeff : WRITEME
        eps : WRITEME
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
        """
        .. todo::

            WRITEME
        """

        rval = OrderedDict()

        if self.scale_by_sharing:
            # scale each learning rate by 1 / # times param is reused
            h_rows, h_cols = self.h_space.shape
            num_h = float(h_rows * h_cols)
            rval[self.transformer._filters] = 1. /num_h
            rval[self.b] = 1. / num_h

        return rval

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return p

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        p,h = total_state

        if not hasattr(self, 'center'):
            self.center = False

        if self.center:
            p -= self.p_offset
            h -= self.h_offset

        return h

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
        W , = self.transformer.get_params()
        return coeffs * T.sqr(W).sum()

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):
        """
        .. todo::

            WRITEME
        """

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
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        self.h_space.validate(downward_state)
        return self.transformer.lmul_T(downward_state)

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        self.transformer.set_batch_size(batch_size)

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        return self.transformer.get_weights_topo()

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME properly

        Returns a shared variable containing an actual state
        (not a mean field state) for this variable.
        """
        raise NotImplementedError("Need to update for C01B")

        t1 = time.time()

        empty_input = self.h_space.get_origin_batch(self.dbm.batch_size)
        h_state = sharedX(empty_input)

        default_z = T.zeros_like(h_state) + self.broadcasted_bias()

        theano_rng = make_theano_rng(None, numpy_rng.randint(2 ** 16),
                                     which_method="binomial")

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

        logger.info('{0}.make_state took {1}'.format(self, t4-t1))
        logger.info('\tcompose time: {0}'.format(t2-t1))
        logger.info('\tcompile time: {0}'.format(t3-t2))
        logger.info('\texecute time: {0}'.format(t4-t3))

        p_state.name = 'p_sample_shared'
        h_state.name = 'h_sample_shared'

        return p_state, h_state

    def expected_energy_term(self, state, average, state_below, average_below):
        """
        .. todo::

            WRITEME
        """

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

    .. todo::

        WRITEME : parameter list
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
        warnings.warn("BVMP_Gaussian math is very faith-based, need to complete gaussian.lyx")

        args = locals()

        del args['input_layer']
        del args['self']
        super(BVMP_Gaussian, self).__init__(**args)
        self.input_layer = input_layer

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError("beta would make get_weights for visualization not correspond to set_weights")
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter = False):
        """
        .. todo::

            WRITEME
        """
        self.b.set_value(biases)
        if recenter:
            assert self.center
            if self.pool_size != 1:
                raise NotImplementedError()
            self.offset.set_value(sigmoid_numpy(self.b.get_value()))

    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        return self.b.get_value() - self.beta_bias().eval()


    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval * self.copies

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size, self.detector_layer_dim).astype(self.b.dtype) + \
                self.b.dimshuffle('x', 0) + self.beta_bias()
        rval = max_pool_channels(z = z,
                pool_size = self.pool_size)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME properly

        Returns a shared variable containing an actual state
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

        theano_rng = make_theano_rng(None, numpy_rng.randint(2 ** 16),
                                     which_method="binomial")

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
        """
        .. todo::

            WRITEME
        """
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
        .. todo::

            WRITEME properly

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
        """
        .. todo::

            WRITEME
        """
        W, = self.transformer.get_params()
        beta = self.input_layer.beta
        assert beta.ndim == 1
        return - 0.5 * T.dot(beta, T.sqr(W))

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):
        """
        .. todo::

            WRITEME
        """

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

class CompositeLayer(HiddenLayer):
    """
        A Layer constructing by aligning several other Layer
        objects side by side

        Parameters
        ----------
        components : WRITEME
            A list of layers that are combined to form this layer
        inputs_to_components : None or dict mapping int to list of int
            Should be None unless the input space is a CompositeSpace
            If inputs_to_components[i] contains j, it means input i will
            be given as input to component j.
            If an input dodes not appear in the dictionary, it will be given
            to all components.

            This field allows one CompositeLayer to have another as input
            without forcing each component to connect to all members
            of the CompositeLayer below. For example, you might want to
            have both densely connected and convolutional units in all
            layers, but a convolutional unit is incapable of taking a
            non-topological input space.
    """


    def __init__(self, layer_name, components, inputs_to_components = None):
        self.layer_name = layer_name

        self.components = list(components)
        assert isinstance(components, list)
        for component in components:
            assert isinstance(component, HiddenLayer)
        self.num_components = len(components)
        self.components = list(components)

        if inputs_to_components is None:
            self.inputs_to_components = None
        else:
            if not isinstance(inputs_to_components, dict):
                raise TypeError("CompositeLayer expected inputs_to_components to be a dict, got "+str(type(inputs_to_components)))
            self.inputs_to_components = OrderedDict()
            for key in inputs_to_components:
                assert isinstance(key, int)
                assert key >= 0
                value = inputs_to_components[key]
                assert isinstance(value, list)
                assert all([isinstance(elem, int) for elem in value])
                assert min(value) >= 0
                assert max(value) < self.num_components
                self.inputs_to_components[key] = list(value)

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

        if not isinstance(space, CompositeSpace):
            assert self.inputs_to_components is None
            self.routing_needed = False
        else:
            if self.inputs_to_components is None:
                self.routing_needed = False
            else:
                self.routing_needed = True
                assert max(self.inputs_to_components) < space.num_components
                # Invert the dictionary
                self.components_to_inputs = OrderedDict()
                for i in xrange(self.num_components):
                    inputs = []
                    for j in xrange(space.num_components):
                        if i in self.inputs_to_components[j]:
                            inputs.append(i)
                    if len(inputs) < space.num_components:
                        self.components_to_inputs[i] = inputs

        for i, component in enumerate(self.components):
            if self.routing_needed and i in self.components_to_inputs:
                cur_space = space.restrict(self.components_to_inputs[i])
            else:
                cur_space = space

            component.set_input_space(cur_space)

        self.output_space = CompositeSpace([ component.get_output_space() for component in self.components ])

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
        return tuple(component.make_state(num_examples, numpy_rng) for
                component in self.components)

    def get_total_state_space(self):
        """
        .. todo::

            WRITEME
        """
        return CompositeSpace([component.get_total_state_space() for component in self.components])

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        for component in self.components:
            component.set_batch_size(batch_size)

    def set_dbm(self, dbm):
        """
        .. todo::

            WRITEME
        """
        for component in self.components:
            component.set_dbm(dbm)

    def mf_update(self, state_below, state_above, layer_above = None, double_weights = False, iter_name = None):
        """
        .. todo::

            WRITEME
        """
        rval = []

        for i, component in enumerate(self.components):
            if self.routing_needed and i in self.components_to_inputs:
                cur_state_below =self.input_space.restrict_batch(state_below, self.components_to_inputs[i])
            else:
                cur_state_below = state_below

            class RoutingLayer(object):
                def __init__(self, idx, layer):
                    self.__dict__.update(locals())
                    del self.self
                    self.layer_name = 'route_'+str(idx)+'_'+layer.layer_name

                def downward_message(self, state):
                    return self.layer.downward_message(state)[self.idx]

            if layer_above is not None:
                cur_layer_above = RoutingLayer(i, layer_above)
            else:
                cur_layer_above = None

            mf_update = component.mf_update(state_below = cur_state_below,
                                            state_above = state_above,
                                            layer_above = cur_layer_above,
                                            double_weights = double_weights,
                                            iter_name = iter_name)

            rval.append(mf_update)

        return tuple(rval)

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
        return tuple([component.init_mf_state() for component in self.components])


    def get_weight_decay(self, coeffs):
        """
        .. todo::

            WRITEME
        """
        return sum([component.get_weight_decay(coeff) for component, coeff
            in safe_zip(self.components, coeffs)])

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return tuple([component.upward_state(elem)
            for component, elem in
            safe_zip(self.components, total_state)])

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return tuple([component.downward_state(elem)
            for component, elem in
            safe_zip(self.components, total_state)])

    def downward_message(self, downward_state):
        """
        .. todo::

            WRITEME
        """
        if isinstance(self.input_space, CompositeSpace):
            num_input_components = self.input_space.num_components
        else:
            num_input_components = 1

        rval = [ None ] * num_input_components

        def add(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return x + y

        for i, packed in enumerate(safe_zip(self.components, downward_state)):
            component, state = packed
            if self.routing_needed and i in self.components_to_inputs:
                input_idx = self.components_to_inputs[i]
            else:
                input_idx = range(num_input_components)

            partial_message = component.downward_message(state)

            if len(input_idx) == 1:
                partial_message = [ partial_message ]

            assert len(input_idx) == len(partial_message)

            for idx, msg in safe_zip(input_idx, partial_message):
                rval[idx] = add(rval[idx], msg)

        if len(rval) == 1:
            rval = rval[0]
        else:
            rval = tuple(rval)

        self.input_space.validate(rval)

        return rval

    def get_l1_act_cost(self, state, target, coeff, eps):
        """
        .. todo::

            WRITEME
        """
        return sum([ comp.get_l1_act_cost(s, t, c, e) \
            for comp, s, t, c, e in safe_zip(self.components, state, target, coeff, eps)])

    def get_range_rewards(self, state, coeffs):
        """
        .. todo::

            WRITEME
        """
        return sum([comp.get_range_rewards(s, c)
            for comp, s, c in safe_zip(self.components, state, coeffs)])

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return reduce(lambda x, y: safe_union(x, y),
                [component.get_params() for component in self.components])

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        logger.info('Get topological weights for which layer?')
        for i, component in enumerate(self.components):
            logger.info('{0} {1}'.format(i, component.layer_name))
        x = raw_input()
        return self.components[int(x)].get_weights_topo()

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """
        rval = OrderedDict()

        for layer, s in safe_zip(self.components, state):
            d = layer.get_monitoring_channels_from_state(s)
            for key in d:
                rval[layer.layer_name+'_'+key] = d[key]

        return rval

    def sample(self, state_below = None, state_above = None,
            layer_above = None,
            theano_rng = None):
        """
        .. todo::

            WRITEME
        """
        rval = []

        for i, component in enumerate(self.components):
            if self.routing_needed and i in self.components_to_inputs:
                cur_state_below =self.input_space.restrict_batch(state_below, self.components_to_inputs[i])
            else:
                cur_state_below = state_below

            class RoutingLayer(object):
                def __init__(self, idx, layer):
                    self.__dict__.update(locals())
                    del self.self
                    self.layer_name = 'route_'+str(idx)+'_'+layer.layer_name

                def downward_message(self, state):
                    return self.layer.downward_message(state)[self.idx]

            if layer_above is not None:
                cur_layer_above = RoutingLayer(i, layer_above)
            else:
                cur_layer_above = None

            sample = component.sample(state_below = cur_state_below,
                                            state_above = state_above,
                                            layer_above = cur_layer_above,
                                            theano_rng = theano_rng)

            rval.append(sample)

        return tuple(rval)
