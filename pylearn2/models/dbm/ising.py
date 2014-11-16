"""
Implementation of a densely connected Ising model in the
pylearn2.models.dbm framework

Notes
-----
If :math:`h` can be -1 or 1, and

.. math::

    p(h) = \exp(T\dot z \dot h),

then the expected value of :math:`h` is given by

.. math::

    \\tanh(T \dot z),

and the probability that :math:`h` is 1 is given by

.. math::

    \sigma(2T \dot z)
"""

__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import operator

import numpy as np

from theano import function
from theano.gof.op import get_debug_values
from theano.compat.six.moves import reduce
from theano.compile.sharedvalue import SharedVariable
import theano.tensor as T
import warnings

from pylearn2.compat import OrderedDict
from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.dbm import init_sigmoid_bias_from_array
from pylearn2.models.dbm.layer import HiddenLayer, VisibleLayer
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng


def init_tanh_bias_from_marginals(dataset, use_y=False):
    """
    .. todo::

        WRITEME
    """
    if use_y:
        X = dataset.y
    else:
        X = dataset.get_design_matrix()
    if not (X.max() == 1):
        raise ValueError("Expected design matrix to consist entirely "
                         "of 0s and 1s, but maximum value is "+str(X.max()))
    assert X.min() == -1.

    mean = X.mean(axis=0)

    mean = np.clip(mean, 1e-7, 1-1e-7)

    init_bias = np.arctanh(mean)

    return init_bias


class IsingVisible(VisibleLayer):
    """
    A DBM visible layer consisting of random variables living
    in a `VectorSpace`, with values in {-1, 1}.

    Implements the energy function term :math:`-\mathbf{b}^T \mathbf{h}`.

    Parameters
    ----------
    nvis : int
        The dimension of the space
    beta : theano shared variable
        Shared variable representing a multiplicative factor of the
        energy function (the inverse temperature)
    learn_beta : boolean, optional
        Whether or not the inverse temperature should be considered as a
        learned parameter
    bias_from_marginals : `pylearn2.datasets.dataset.Dataset`, optional
        A dataset whose marginals are used to initialize the visible
        biases
    """

    def __init__(self, nvis, beta, learn_beta=False, bias_from_marginals=None):
        if not isinstance(beta, SharedVariable):
            raise ValueError("beta needs to be a theano shared variable.")
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
            init_bias = init_tanh_bias_from_marginals(bias_from_marginals)

        self.bias = sharedX(init_bias, 'visible_bias')

    def get_biases(self):
        """
        .. todo::

            WRITEME
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
        return total_state

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        rval = [self.bias]
        if self.learn_beta:
            rval.append(self.beta)
        return rval

    def mf_update(self, state_above, layer_above):
        """
        .. todo::

            WRITEME
        """
        msg = layer_above.downward_message(state_above)

        bias = self.bias

        z = msg + bias

        rval = T.tanh(self.beta * z)

        return rval

    def sample(self, state_below=None, state_above=None, layer_above=None,
               theano_rng=None):
        """
        .. todo::

            WRITEME
        """

        assert state_below is None

        msg = layer_above.downward_message(state_above)

        bias = self.bias

        z = msg + bias

        phi = T.nnet.sigmoid(2. * self.beta * z)

        rval = theano_rng.binomial(size=phi.shape, p=phi, dtype=phi.dtype, n=1)

        return rval * 2. - 1.

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
        driver = numpy_rng.uniform(0., 1., (num_examples, self.nvis))
        on_prob = sigmoid_numpy(2. * self.beta.get_value() *
                                self.bias.get_value())
        sample = 2. * (driver < on_prob) - 1.

        rval = sharedX(sample, name='v_sample_shared')

        return rval

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        .. todo::

            WRITEME
        """
        mean = T.nnet.sigmoid(2. * self.beta * self.b)
        rval = theano_rng.binomial(size=(num_examples, self.nvis), p=mean)
        rval = 2. * (rval) - 1.

        return rval

    def expected_energy_term(self, state, average, state_below=None,
                             average_below=None):
        """
        .. todo::

            WRITEME
        """
        assert state_below is None
        assert average_below is None
        assert average in [True, False]
        self.space.validate(state)

        # Energy function is linear so it doesn't matter if we're averaging
        # or not
        rval = -(self.beta * T.dot(state, self.bias))

        assert rval.ndim == 1

        return rval


class IsingHidden(HiddenLayer):
    """
    A hidden layer with :math:`\mathbf{h}` being a vector in {-1, 1},
    implementing the energy function term

    .. math::

        -\mathbf{v}^T \mathbf{W}\mathbf{h} -\mathbf{b}^T \mathbf{h}

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are parameters of this
    layer, and :math:`\mathbf{v}` is the upward state of the layer below.

    Parameters
    ----------
    dim : WRITEME
    layer_name : WRITEME
    beta : theano shared variable
        Shared variable representing a multiplicative factor of the energy
        function (the inverse temperature)
    learn_beta : boolean, optional
        Whether or not the inverse temperature should be considered as a
        learned parameter
    irange : WRITEME
    sparse_init : WRITEME
    sparse_stdev : WRITEME
    include_prob : float, optional
        Probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is
        initialized to 0.
    init_bias : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_col_norm : WRITEME
    """

    def __init__(self,
                 dim,
                 layer_name,
                 beta,
                 learn_beta=False,
                 irange=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_col_norm=None):
        if not isinstance(beta, SharedVariable):
            raise ValueError("beta needs to be a theano shared variable.")
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX(np.zeros((self.dim,)) + init_bias,
                         name=layer_name + '_b')

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

            WRITEME properly

        Notes
        -----
        Note: this resets parameters!
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        rng = self.dbm.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange, self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                    < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

    def _modify_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

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
        return VectorSpace(self.dim)

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        if self.learn_beta:
            rval.append(self.beta)
        return rval

    def get_weight_decay(self, coeff):
        """
        .. todo::

            WRITEME
        """
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
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
        W, = self.transformer.get_params()
        return W.get_value()

    def set_weights(self, weights):
        """
        .. todo::

            WRITEME
        """
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases, recenter=False):
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

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W, = self.transformer.get_params()

        W = W.T

        W = W.reshape(
            (self.detector_layer_dim, self.input_space.shape[0],
             self.input_space.shape[1], self.input_space.nchannels)
        )

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
            ('row_norms_min', row_norms.min()),
            ('row_norms_mean', row_norms.mean()),
            ('row_norms_max', row_norms.max()),
            ('col_norms_min', col_norms.min()),
            ('col_norms_mean', col_norms.mean()),
            ('col_norms_max', col_norms.max()),
        ])

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples". The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
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
                ('mean_x.min_u', v_mean.min()),
            ]:
                rval[prefix+key] = val

        return rval

    def sample(self, state_below=None, state_above=None, layer_above=None,
               theano_rng=None):
        """
        .. todo::

            WRITEME
        """

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to " +
                             "None so that it may appear after layer_above " +
                             "/ state_above in the list.")

        if state_above is not None:
            msg = layer_above.downward_message(state_above)
        else:
            msg = None

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if msg is not None:
            z = z + msg

        on_prob = T.nnet.sigmoid(2. * self.beta * z)

        samples = theano_rng.binomial(p=on_prob, n=1, size=on_prob.shape,
                                      dtype=on_prob.dtype) * 2. - 1.

        return samples

    def downward_message(self, downward_state):
        """
        .. todo::

            WRITEME
        """
        rval = self.transformer.lmul_T(downward_state)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size,
                    self.dim).astype(self.b.dtype) + \
            self.b.dimshuffle('x', 0)
        rval = T.tanh(self.beta * z)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME properly

        Returns a shared variable containing an actual state
        (not a mean field state) for this variable.
        """
        driver = numpy_rng.uniform(0., 1., (num_examples, self.dim))
        on_prob = sigmoid_numpy(2. * self.beta.get_value() *
                                self.b.get_value())
        sample = 2. * (driver < on_prob) - 1.

        rval = sharedX(sample, name='v_sample_shared')

        return rval

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        .. todo::

            WRITEME
        """
        mean = T.nnet.sigmoid(2. * self.beta * self.b)
        rval = theano_rng.binomial(size=(num_examples, self.nvis), p=mean)
        rval = 2. * (rval) - 1.

        return rval

    def expected_energy_term(self, state, average, state_below, average_below):
        """
        .. todo::

            WRITEME
        """

        # state = Print('h_state', attrs=['min', 'max'])(state)

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got " +
                                         "shape of %d" % (self.dbm.batch_size,
                                                          sb.shape[0]))
                    assert reduce(operator.mul, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        # Energy function is linear so it doesn't matter if we're averaging or
        # not. Specifically, our terms are -u^T W d - b^T d where u is the
        # upward state of layer below and d is the downward state of this layer

        bias_term = T.dot(state, self.b)
        weights_term = (self.transformer.lmul(state_below) * state).sum(axis=1)

        rval = -bias_term - weights_term

        rval *= self.beta

        assert rval.ndim == 1

        return rval

    def linear_feed_forward_approximation(self, state_below):
        """
        .. todo::

            WRITEME properly

        Used to implement TorontoSparsity. Unclear exactly what properties of
        it are important or how to implement it for other layers.

        Properties it must have:
            output is same kind of data structure (ie, tuple of theano
            2-tensors) as mf_update

        Properties it probably should have for other layer types:
            An infinitesimal change in state_below or the parameters should
            cause the same sign of change in the output of
            linear_feed_forward_approximation and in mf_update

            Should not have any non-linearities that cause the gradient to
            shrink

            Should disregard top-down feedback
        """

        z = self.beta * (self.transformer.lmul(state_below) + self.b)

        if self.pool_size != 1:
            # Should probably implement sum pooling for the non-pooled version,
            # but in reality it's not totally clear what the right answer is
            raise NotImplementedError()

        return z, z

    def mf_update(self, state_below, state_above, layer_above=None,
                  double_weights=False, iter_name=None):
        """
        .. todo::

            WRITEME
        """

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got " +
                                         "shape of %d" % (self.dbm.batch_size,
                                                          sb.shape[0]))
                    assert reduce(operator.mul, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_' + layer_above.layer_name + '_to_' + \
                       self.layer_name + '[' + iter_name + ']'
        else:
            msg = None

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        if msg is not None:
            z = z + msg
        h = T.tanh(self.beta * z)

        return h


class BoltzmannIsingVisible(VisibleLayer):
    """
    An IsingVisible whose parameters are defined in Boltzmann machine space.

    Notes
    -----
    All parameter noise/clipping is handled by BoltzmannIsingHidden.

    .. todo::

        WRITEME properly

    Parameters
    ----------
    nvis : int
        Number of visible units
    beta : theano shared variable
        Shared variable representing a multiplicative factor of the energy
        function (the inverse temperature)
    learn_beta : boolean, optional
        Whether or not the inverse temperature should be considered
            as a learned parameter
    bias_from_marginals : `pylearn2.datasets.dataset.Dataset`, optional
        A dataset whose marginals are used to initialize the visible
        biases
    sampling_b_stdev : WRITEME
    min_ising_b : WRITEME
    max_ising_b : WRITEME
    """

    def __init__(self, nvis, beta, learn_beta=False, bias_from_marginals=None,
                 sampling_b_stdev=None, min_ising_b=None, max_ising_b=None):

        if not isinstance(beta, SharedVariable):
            raise ValueError("beta needs to be a theano shared " +
                             "variable.")
        self.__dict__.update(locals())
        del self.self
        # Don't serialize the dataset
        del self.bias_from_marginals

        self.space = VectorSpace(nvis)
        self.input_space = self.space

        if bias_from_marginals is None:
            init_bias = np.zeros((nvis,))
        else:
            # data is in [-1, 1], but want biases for a sigmoid
            init_bias = \
                init_sigmoid_bias_from_array(bias_from_marginals.X / 2. + 0.5)
            # init_bias =
        self.boltzmann_bias = sharedX(init_bias, 'visible_bias')

        self.resample_fn = None

    def finalize_initialization(self):
        """
        .. todo::

            WRITEME
        """
        if self.sampling_b_stdev is not None:
            self.noisy_sampling_b = \
                sharedX(np.zeros((self.layer_above.dbm.batch_size, self.nvis)))

        updates = OrderedDict()
        updates[self.boltzmann_bias] = self.boltzmann_bias
        updates[self.layer_above.W] = self.layer_above.W
        self.enforce_constraints()

    def _modify_updates(self, updates):
        """
        .. todo::

            WRITEME
        """
        beta = self.beta
        if beta in updates:
            updated_beta = updates[beta]
            updates[beta] = T.clip(updated_beta, 1., 1000.)

        if any(constraint is not None for constraint in [self.min_ising_b,
                                                         self.max_ising_b]):
            bmn = self.min_ising_b
            if bmn is None:
                bmn = - 1e6
            bmx = self.max_ising_b
            if bmx is None:
                bmx = 1e6
            wmn_above = self.layer_above.min_ising_W
            if wmn_above is None:
                wmn_above = - 1e6
            wmx_above = self.layer_above.max_ising_W
            if wmx_above is None:
                wmx_above = 1e6

            b = updates[self.boltzmann_bias]
            W_above = updates[self.layer_above.W]
            ising_b = 0.5 * b + 0.25 * W_above.sum(axis=1)
            ising_b = T.clip(ising_b, bmn, bmx)

            ising_W_above = 0.25 * W_above
            ising_W_above = T.clip(ising_W_above, wmn_above, wmx_above)
            bhn = 2. * (ising_b - ising_W_above.sum(axis=1))

            updates[self.boltzmann_bias] = bhn

        if self.noisy_sampling_b is not None:
            theano_rng = make_theano_rng(None, self.dbm.rng.randint(2**16), which_method="normal")

            b = updates[self.boltzmann_bias]
            W_above = updates[self.layer_above.W]
            ising_b = 0.5 * b + 0.25 * W_above.sum(axis=1)

            noisy_sampling_b = \
                theano_rng.normal(avg=ising_b.dimshuffle('x', 0),
                                  std=self.sampling_b_stdev,
                                  size=self.noisy_sampling_b.shape,
                                  dtype=ising_b.dtype)
            updates[self.noisy_sampling_b] = noisy_sampling_b

    def resample_bias_noise(self, batch_size_changed=False):
        """
        .. todo::

            WRITEME
        """
        if batch_size_changed:
            self.resample_fn = None

        if self.resample_fn is None:
            updates = OrderedDict()

            if self.sampling_b_stdev is not None:
                self.noisy_sampling_b = \
                    sharedX(np.zeros((self.dbm.batch_size, self.nvis)))

            if self.noisy_sampling_b is not None:
                theano_rng = make_theano_rng(None, self.dbm.rng.randint(2**16), which_method="normal")

                b = self.boltzmann_bias
                W_above = self.layer_above.W
                ising_b = 0.5 * b + 0.25 * W_above.sum(axis=1)

                noisy_sampling_b = \
                    theano_rng.normal(avg=ising_b.dimshuffle('x', 0),
                                      std=self.sampling_b_stdev,
                                      size=self.noisy_sampling_b.shape,
                                      dtype=ising_b.dtype)
                updates[self.noisy_sampling_b] = noisy_sampling_b

            self.resample_fn = function([], updates=updates)

        self.resample_fn()

    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("BoltzmannIsingVisible.get_biases returns the " +
                      "BOLTZMANN biases, is that what we want?")
        return self.boltzmann_bias.get_value()

    def set_biases(self, biases, recenter=False):
        """
        .. todo::

            WRITEME
        """
        assert False  # not really sure what this should do for this layer

    def ising_bias(self, for_sampling=False):
        """
        .. todo::

            WRITEME
        """
        if for_sampling and self.layer_above.sampling_b_stdev is not None:
            return self.noisy_sampling_b
        return \
            0.5 * self.boltzmann_bias + 0.25 * self.layer_above.W.sum(axis=1)

    def ising_bias_numpy(self):
        """
        .. todo::

            WRITEME
        """
        return 0.5 * self.boltzmann_bias.get_value() + \
            0.25 * self.layer_above.W.get_value().sum(axis=1)

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        rval = [self.boltzmann_bias]
        if self.learn_beta:
            rval.append(self.beta)
        return rval

    def sample(self, state_below=None, state_above=None, layer_above=None,
               theano_rng=None):
        """
        .. todo::

            WRITEME
        """

        assert state_below is None

        msg = layer_above.downward_message(state_above, for_sampling=True)

        bias = self.ising_bias(for_sampling=True)

        z = msg + bias

        phi = T.nnet.sigmoid(2. * self.beta * z)

        rval = theano_rng.binomial(size=phi.shape, p=phi, dtype=phi.dtype, n=1)

        return rval * 2. - 1.

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME
        """
        driver = numpy_rng.uniform(0., 1., (num_examples, self.nvis))
        on_prob = sigmoid_numpy(2. * self.beta.get_value() *
                                self.ising_bias_numpy())
        sample = 2. * (driver < on_prob) - 1.

        rval = sharedX(sample, name='v_sample_shared')

        return rval

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        .. todo::

            WRITEME
        """
        mean = T.nnet.sigmoid(2. * self.beta * self.ising_bias())
        rval = theano_rng.binomial(size=(num_examples, self.nvis), p=mean)
        rval = 2. * (rval) - 1.

        return rval

    def mf_update(self, state_above, layer_above):
        """
        .. todo::

            WRITEME
        """
        msg = layer_above.downward_message(state_above, for_sampling=True)

        bias = self.ising_bias(for_sampling=True)

        z = msg + bias

        rval = T.tanh(self.beta * z)

        return rval

    def expected_energy_term(self, state, average, state_below=None,
                             average_below=None):
        """
        .. todo::

            WRITEME
        """

        # state = Print('v_state', attrs=['min', 'max'])(state)

        assert state_below is None
        assert average_below is None
        assert average in [True, False]
        self.space.validate(state)

        # Energy function is linear so it doesn't matter if we're averaging
        # or not
        rval = -(self.beta * T.dot(state, self.ising_bias()))

        assert rval.ndim == 1

        return rval

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """
        rval = OrderedDict()

        ising_b = self.ising_bias()

        rval['ising_b_min'] = ising_b.min()
        rval['ising_b_max'] = ising_b.max()
        rval['beta'] = self.beta

        if hasattr(self, 'noisy_sampling_b'):
            rval['noisy_sampling_b_min'] = self.noisy_sampling_b.min()
            rval['noisy_sampling_b_max'] = self.noisy_sampling_b.max()

        return rval


class BoltzmannIsingHidden(HiddenLayer):
    """
    An IsingHidden whose parameters are defined in Boltzmann machine space.

    .. todo::

        WRITEME properly

    Parameters
    ----------
    dim : WRITEME
    layer_name : WRITEME
    layer_below : WRITEME
    beta : theano shared variable
        Shared variable representing a multiplicative factor of the energy
        function (the inverse temperature)
    learn_beta : boolean, optional
        Whether or not the inverse temperature should be considered as a
        learned parameter
    irange : WRITEME
    sparse_init : WRITEME
    sparse_stdev : WRITEME
    include_prob : WRITEME
    init_bias : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    beta_lr_scale : WRITEME
    max_col_norm : WRITEME
    min_ising_b : WRITEME
    max_ising_b : WRITEME
    min_ising_W : WRITEME
    max_ising_W : WRITEME
    sampling_W_stdev : WRITEME
    sampling_b_stdev : WRITEME
    """
    def __init__(self,
                 dim,
                 layer_name,
                 layer_below,
                 beta,
                 learn_beta=False,
                 irange=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 beta_lr_scale=None,
                 max_col_norm=None,
                 min_ising_b=None,
                 max_ising_b=None,
                 min_ising_W=None,
                 max_ising_W=None,
                 sampling_W_stdev=None,
                 sampling_b_stdev=None):
        if not isinstance(beta, SharedVariable):
            raise ValueError("beta needs to be a theano shared variable.")
        self.__dict__.update(locals())
        del self.self

        layer_below.layer_above = self
        self.layer_above = None
        self.resample_fn = None

    def get_lr_scalers(self):
        """
        .. todo::

            WRITEME
        """

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if not hasattr(self, 'beta_lr_scale'):
            self.beta_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W = self.W
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.boltzmann_b] = self.b_lr_scale

        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale

        return rval

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME properly

        Note: this resets parameters!
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        rng = self.dbm.rng

        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange, self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                    < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))
            W *= self.sparse_stdev
        W = sharedX(W)
        W.name = self.layer_name + '_W'
        self.W = W

        self.boltzmann_b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                                   name=self.layer_name + '_b')

    def finalize_initialization(self):
        """
        .. todo::

            WRITEME
        """
        if self.sampling_b_stdev is not None:
            self.noisy_sampling_b = \
                sharedX(np.zeros((self.dbm.batch_size, self.dim)))
        if self.sampling_W_stdev is not None:
            self.noisy_sampling_W = \
                sharedX(np.zeros((self.input_dim, self.dim)),
                        'noisy_sampling_W')

        updates = OrderedDict()
        updates[self.boltzmann_b] = self.boltzmann_b
        updates[self.W] = self.W
        if self.layer_above is not None:
            updates[self.layer_above.W] = self.layer_above.W
        self.enforce_constraints()

    def _modify_updates(self, updates):
        """
        .. todo::

            WRITEME
        """
        beta = self.beta
        if beta in updates:
            updated_beta = updates[beta]
            updates[beta] = T.clip(updated_beta, 1., 1000.)

        if self.max_col_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        if any(constraint is not None for constraint in [self.min_ising_b,
                                                         self.max_ising_b,
                                                         self.min_ising_W,
                                                         self.max_ising_W]):
            bmn = self.min_ising_b
            if bmn is None:
                bmn = - 1e6
            bmx = self.max_ising_b
            if bmx is None:
                bmx = 1e6
            wmn = self.min_ising_W
            if wmn is None:
                wmn = - 1e6
            wmx = self.max_ising_W
            if wmx is None:
                wmx = 1e6
            if self.layer_above is not None:
                wmn_above = self.layer_above.min_ising_W
                if wmn_above is None:
                    wmn_above = - 1e6
                wmx_above = self.layer_above.max_ising_W
                if wmx_above is None:
                    wmx_above = 1e6

            W = updates[self.W]
            ising_W = 0.25 * W
            ising_W = T.clip(ising_W, wmn, wmx)

            b = updates[self.boltzmann_b]
            if self.layer_above is not None:
                W_above = updates[self.layer_above.W]
                ising_b = 0.5 * b + 0.25 * W.sum(axis=0) \
                                  + 0.25 * W_above.sum(axis=1)
            else:
                ising_b = 0.5 * b + 0.25 * W.sum(axis=0)
            ising_b = T.clip(ising_b, bmn, bmx)

            if self.layer_above is not None:
                ising_W_above = 0.25 * W_above
                ising_W_above = T.clip(ising_W_above, wmn_above, wmx_above)
                bhn = 2. * (ising_b - ising_W.sum(axis=0)
                                    - ising_W_above.sum(axis=1))
            else:
                bhn = 2. * (ising_b - ising_W.sum(axis=0))
            Wn = 4. * ising_W

            updates[self.W] = Wn
            updates[self.boltzmann_b] = bhn

        if self.noisy_sampling_W is not None:
            theano_rng = make_theano_rng(None, self.dbm.rng.randint(2**16), which_method="normal")

            W = updates[self.W]
            ising_W = 0.25 * W

            noisy_sampling_W = \
                theano_rng.normal(avg=ising_W, std=self.sampling_W_stdev,
                                  size=ising_W.shape, dtype=ising_W.dtype)
            updates[self.noisy_sampling_W] = noisy_sampling_W

            b = updates[self.boltzmann_b]
            if self.layer_above is not None:
                W_above = updates[self.layer_above.W]
                ising_b = 0.5 * b + 0.25 * W.sum(axis=0) \
                                  + 0.25 * W_above.sum(axis=1)
            else:
                ising_b = 0.5 * b + 0.25 * W.sum(axis=0)

            noisy_sampling_b = \
                theano_rng.normal(avg=ising_b.dimshuffle('x', 0),
                                  std=self.sampling_b_stdev,
                                  size=self.noisy_sampling_b.shape,
                                  dtype=ising_b.dtype)
            updates[self.noisy_sampling_b] = noisy_sampling_b

    def resample_bias_noise(self, batch_size_changed=False):
        """
        .. todo::

            WRITEME
        """
        if batch_size_changed:
            self.resample_fn = None

        if self.resample_fn is None:
            updates = OrderedDict()

            if self.sampling_b_stdev is not None:
                self.noisy_sampling_b = \
                    sharedX(np.zeros((self.dbm.batch_size, self.dim)))

            if self.noisy_sampling_b is not None:
                theano_rng = make_theano_rng(None, self.dbm.rng.randint(2**16), which_method="normal")

                b = self.boltzmann_b
                if self.layer_above is not None:
                    W_above = self.layer_above.W
                    ising_b = 0.5 * b + 0.25 * self.W.sum(axis=0) \
                                      + 0.25 * W_above.sum(axis=1)
                else:
                    ising_b = 0.5 * b + 0.25 * self.W.sum(axis=0)

                noisy_sampling_b = \
                    theano_rng.normal(avg=ising_b.dimshuffle('x', 0),
                                      std=self.sampling_b_stdev,
                                      size=self.noisy_sampling_b.shape,
                                      dtype=ising_b.dtype)
                updates[self.noisy_sampling_b] = noisy_sampling_b

            self.resample_fn = function([], updates=updates)

        self.resample_fn()

    def get_total_state_space(self):
        """
        .. todo::

            WRITEME
        """
        return VectorSpace(self.dim)

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.boltzmann_b.name is not None
        W = self.W
        assert W.name is not None
        rval = [W]
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.boltzmann_b not in rval
        rval.append(self.boltzmann_b)
        if self.learn_beta:
            rval.append(self.beta)
        return rval

    def ising_weights(self, for_sampling=False):
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'sampling_W_stdev'):
            self.sampling_W_stdev = None
        if for_sampling and self.sampling_W_stdev is not None:
            return self.noisy_sampling_W
        return 0.25 * self.W

    def ising_b(self, for_sampling=False):
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'sampling_b_stdev'):
            self.sampling_b_stdev = None
        if for_sampling and self.sampling_b_stdev is not None:
            return self.noisy_sampling_b
        else:
            if self.layer_above is not None:
                return 0.5 * self.boltzmann_b + \
                    0.25 * self.W.sum(axis=0) + \
                    0.25 * self.layer_above.W.sum(axis=1)
            else:
                return 0.5 * self.boltzmann_b + 0.25 * self.W.sum(axis=0)

    def ising_b_numpy(self):
        """
        .. todo::

            WRITEME
        """
        if self.layer_above is not None:
            return 0.5 * self.boltzmann_b.get_value() + \
                0.25 * self.W.get_value().sum(axis=0) + \
                0.25 * self.layer_above.W.get_value().sum(axis=1)
        else:
            return 0.5 * self.boltzmann_b.get_value() + \
                0.25 * self.W.get_value().sum(axis=0)

    def get_weight_decay(self, coeff):
        """
        .. todo::

            WRITEME
        """
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * T.sqr(W).sum()

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("BoltzmannIsingHidden.get_weights returns the " +
                      "BOLTZMANN weights, is that what we want?")
        W = self.W
        return W.get_value()

    def set_weights(self, weights):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("BoltzmannIsingHidden.set_weights sets the BOLTZMANN " +
                      "weights, is that what we want?")
        W = self.W
        W.set_value(weights)

    def set_biases(self, biases, recenter=False):
        """
        .. todo::

            WRITEME
        """
        self.boltzmann_b.set_value(biases)
        assert not recenter  # not really sure what this should do if True

    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("BoltzmannIsingHidden.get_biases returns the " +
                      "BOLTZMANN biases, is that what we want?")
        return self.boltzmann_b.get_value()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ('v', 'h')

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("BoltzmannIsingHidden.get_weights_topo returns the " +
                      "BOLTZMANN weights, is that what we want?")

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W = self.W

        W = W.T

        W = W.reshape((self.detector_layer_dim, self.input_space.shape[0],
                       self.input_space.shape[1], self.input_space.nchannels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def upward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def downward_state(self, total_state):
        """
        .. todo::

            WRITEME
        """
        return total_state

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([
            ('boltzmann_row_norms_min', row_norms.min()),
            ('boltzmann_row_norms_mean', row_norms.mean()),
            ('boltzmann_row_norms_max', row_norms.max()),
            ('boltzmann_col_norms_min', col_norms.min()),
            ('boltzmann_col_norms_mean', col_norms.mean()),
            ('boltzmann_col_norms_max', col_norms.max()),
        ])

        ising_W = self.ising_weights()

        rval['ising_W_min'] = ising_W.min()
        rval['ising_W_max'] = ising_W.max()

        ising_b = self.ising_b()

        rval['ising_b_min'] = ising_b.min()
        rval['ising_b_max'] = ising_b.max()

        if hasattr(self, 'noisy_sampling_W'):
            rval['noisy_sampling_W_min'] = self.noisy_sampling_W.min()
            rval['noisy_sampling_W_max'] = self.noisy_sampling_W.max()
            rval['noisy_sampling_b_min'] = self.noisy_sampling_b.min()
            rval['noisy_sampling_b_max'] = self.noisy_sampling_b.max()

        return rval

    def get_monitoring_channels_from_state(self, state):
        """
        .. todo::

            WRITEME
        """

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples". The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
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

    def sample(self, state_below=None, state_above=None, layer_above=None,
               theano_rng=None):
        """
        .. todo::

            WRITEME
        """

        if theano_rng is None:
            raise ValueError("theano_rng is required; it just defaults to " +
                             "None so that it may appear after layer_above " +
                             "/ state_above in the list.")

        if state_above is not None:
            msg = layer_above.downward_message(state_above, for_sampling=True)
        else:
            msg = None

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = T.dot(state_below, self.ising_weights(for_sampling=True)) + \
            self.ising_b(for_sampling=True)

        if msg is not None:
            z = z + msg

        on_prob = T.nnet.sigmoid(2. * self.beta * z)

        samples = theano_rng.binomial(p=on_prob, n=1, size=on_prob.shape,
                                      dtype=on_prob.dtype) * 2. - 1.

        return samples

    def downward_message(self, downward_state, for_sampling=False):
        """
        .. todo::

            WRITEME
        """
        rval = T.dot(downward_state,
                     self.ising_weights(for_sampling=for_sampling).T)

        if self.requires_reformat:
            rval = self.desired_space.format_as(rval, self.input_space)

        return rval

    def init_mf_state(self):
        """
        .. todo::

            WRITEME
        """
        # work around theano bug with broadcasted vectors
        z = T.alloc(0., self.dbm.batch_size,
                    self.dim).astype(self.boltzmann_b.dtype) + \
            self.ising_b().dimshuffle('x', 0)
        rval = T.tanh(self.beta * z)
        return rval

    def make_state(self, num_examples, numpy_rng):
        """
        .. todo::

            WRITEME properly

        Returns a shared variable containing an actual state
        (not a mean field state) for this variable.
        """
        driver = numpy_rng.uniform(0., 1., (num_examples, self.dim))
        on_prob = sigmoid_numpy(2. * self.beta.get_value() *
                                self.ising_b_numpy())
        sample = 2. * (driver < on_prob) - 1.

        rval = sharedX(sample, name='v_sample_shared')

        return rval

    def make_symbolic_state(self, num_examples, theano_rng):
        """
        .. todo::

            WRITEME
        """
        mean = T.nnet.sigmoid(2. * self.beta * self.ising_b())
        rval = theano_rng.binomial(size=(num_examples, self.dim), p=mean)
        rval = 2. * (rval) - 1.

        return rval

    def expected_energy_term(self, state, average, state_below, average_below):
        """
        .. todo::

            WRITEME
        """

        # state = Print('h_state', attrs=['min', 'max'])(state)

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got " +
                                         "shape of %d" % (self.dbm.batch_size,
                                                          sb.shape[0]))
                    assert reduce(operator.mul, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        # Energy function is linear so it doesn't matter if we're averaging or
        # not. Specifically, our terms are -u^T W d - b^T d where u is the
        # upward state of layer below and d is the downward state of this layer

        bias_term = T.dot(state, self.ising_b())
        weights_term = \
            (T.dot(state_below, self.ising_weights()) * state).sum(axis=1)

        rval = -bias_term - weights_term

        rval *= self.beta

        assert rval.ndim == 1

        return rval

    def linear_feed_forward_approximation(self, state_below):
        """
        .. todo::

            WRITEME properly

        Used to implement TorontoSparsity. Unclear exactly what properties of
        it are important or how to implement it for other layers.

        Properties it must have:
            output is same kind of data structure (ie, tuple of theano
            2-tensors) as mf_update

        Properties it probably should have for other layer types:
            An infinitesimal change in state_below or the parameters should
            cause the same sign of change in the output of
            linear_feed_forward_approximation and in mf_update

            Should not have any non-linearities that cause the gradient to
            shrink

            Should disregard top-down feedback
        """

        z = self.beta * (T.dot(state_below, self.ising_weights()) + self.ising_b())

        return z

    def mf_update(self, state_below, state_above, layer_above=None,
                  double_weights=False, iter_name=None):
        """
        .. todo::

            WRITEME
        """

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got " +
                                         "shape of %d" % (self.dbm.batch_size,
                                                          sb.shape[0]))
                    assert reduce(operator.mul, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        if iter_name is None:
            iter_name = 'anon'

        if state_above is not None:
            assert layer_above is not None
            msg = layer_above.downward_message(state_above)
            msg.name = 'msg_from_' + layer_above.layer_name + '_to_' + \
                       self.layer_name + '[' + iter_name+']'
        else:
            msg = None

        if double_weights:
            state_below = 2. * state_below
            state_below.name = self.layer_name + '_'+iter_name + '_2state'
        z = T.dot(state_below, self.ising_weights()) + self.ising_b()
        if self.layer_name is not None and iter_name is not None:
            z.name = self.layer_name + '_' + iter_name + '_z'
        if msg is not None:
            z = z + msg
        h = T.tanh(self.beta * z)

        return h

    def get_l2_act_cost(self, state, target, coeff):
        """
        .. todo::

            WRITEME
        """
        avg = state.mean(axis=0)
        diff = avg - target
        return coeff * T.sqr(diff).mean()
