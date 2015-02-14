"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
from pylearn2.compat import OrderedDict
from pylearn2.models import mlp
from pylearn2.models.mlp import Layer
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.space import CompositeSpace
from pylearn2.space import Space
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps

from pylearn2.expr.nnet import arg_of_sigmoid
from pylearn2.space import Conv2DSpace
from pylearn2.utils import py_integer_types

import theano
from theano import tensor as T
from theano import config

import numpy as np


class Softmax(mlp.Softmax):
    """
    An extension of the MLP's softmax layer which monitors
    the perplexity

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    """
    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      target=None):

        mx = state.max(axis=1)

        rval = OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())])

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'] / T.log(2))

        return rval


class ProjectionLayer(Layer):
    """
    This layer can be used to project discrete labels into a continous space
    as done in e.g. language models. It takes labels as an input (IndexSpace)
    and maps them to their continous embeddings and concatenates them.

    Parameters
        ----------
    dim : int
        The dimension of the embeddings. Note that this means that the
        output dimension is (dim * number of input labels)
    layer_name : string
        Layer name
    irange : numeric
       The range of the uniform distribution used to initialize the
       embeddings. Can't be used with istdev.
    istdev : numeric
        The standard deviation of the normal distribution used to
        initialize the embeddings. Can't be used with irange.
    """
    def __init__(self, dim, layer_name, irange=None, istdev=None):
        """
        Initializes a projection layer.
        """
        super(ProjectionLayer, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        if irange is None and istdev is None:
            raise ValueError("ProjectionLayer needs either irange or"
                             "istdev in order to intitalize the projections.")
        elif irange is not None and istdev is not None:
            raise ValueError("ProjectionLayer was passed both irange and "
                             "istdev but needs only one")
        else:
            self._irange = irange
            self._istdev = istdev

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, *args, **kwargs):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min', row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max', row_norms.max()),
                            ('col_norms_min', col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max', col_norms.max()), ])

    def _check_input_space_and_get_max_labels(self, space):
        if isinstance(space, IndexSpace):
            return space.max_labels
        if isinstance(space, CompositeSpace):
            ml = []
            for c in space.components:
                ml.append(self._check_input_space_and_get_max_labels(c))
            # check that all of them are equal
            if len(set(ml)) != 1:
                raise ValueError("Composite space is empty or containing "
                                 "incompatible index spaces")
            return ml[0]
        raise ValueError("ProjectionLayer needs an IndexSpace or a "
                         "CompositeSpace of them as input")

    def _build_output_space(self, space):
        if isinstance(space, IndexSpace):
            return VectorSpace(self.dim * space.dim)
        if isinstance(space, CompositeSpace):
            return CompositeSpace([self._build_output_space(c)
                                   for c in space.components])
        assert False

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        max_labels = self._check_input_space_and_get_max_labels(space)
        self.input_space = space
        self.output_space = self._build_output_space(space)
        rng = self.mlp.rng
        if self._irange is not None:
            W = rng.uniform(-self._irange,
                            self._irange,
                            (max_labels, self.dim))
        else:
            W = rng.randn(max_labels, self.dim) * self._istdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

    def _fprop_recursive(self, state_below):
        if isinstance(state_below, tuple):
            return tuple(self._fprop_recursive(s) for s in state_below)
        return self.transformer.project(state_below)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        return self._fprop_recursive(state_below)

    @wraps(Layer.get_params)
    def get_params(self):
        W, = self.transformer.get_params()
        assert W.name is not None
        params = [W]
        return params

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()


class ContrastiveProbabilityLayer(Layer):
    """
    Predicts the probability of a sample to come from a positive distribution
    or a negative distribution. For each sample (i.e. a row in the input),
    this layer produces *two* sets of probabilities. The first set has n
    probabilities which says "if the sample is of class i then the probability
    it comes from the positive distribution is p_i". The second set is the
    complement of the first.

    This layer should be provided with *two* targets, one positive and one
    negative (randomly generated). To improve performance one should use
    binary targets in which case the spaces should be index spaces.

    It can be used to perform negative sampling.
    """

    def __init__(self, n_classes, layer_name, irange=None, sparse_init=None,
                 istdev=None, binary_target_dim=None, no_affine=False):
        super(ContrastiveProbabilityLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self

        if binary_target_dim is not None:
            assert isinstance(binary_target_dim, py_integer_types)
            self._has_binary_target = True
            indiv_space = IndexSpace(dim=binary_target_dim,
                                     max_labels=n_classes)
        else:
            self._has_binary_target = False
            indiv_space = VectorSpace(n_classes)
        self._target_space = CompositeSpace((indiv_space, indiv_space))
        self.output_space = CompositeSpace((VectorSpace(n_classes),
                                            VectorSpace(n_classes)))
        if not no_affine:
            self.b = sharedX(np.zeros((n_classes,)), name='contrast_b')

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space) + " of type " + str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.no_affine:
            self._params = []
        else:
            num_cols = self.n_classes
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W = rng.uniform(-self.irange,
                                self.irange,
                                (self.input_dim, num_cols))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W = rng.randn(self.input_dim, num_cols) * self.istdev
            else:
                assert self.sparse_init is not None
                W = np.zeros((self.input_dim, num_cols))
                for i in xrange(num_cols):
                    for _ in xrange(self.sparse_init):
                        idx = rng.randint(0, self.input_dim)
                        while W[idx, i] != 0.:
                            idx = rng.randint(0, self.input_dim)
                        W[idx, i] = rng.randn()

            self.W = sharedX(W, 'contrast_W')

            self._params = [self.b, self.W]

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            Z = T.dot(state_below, self.W) + self.b

        p = T.nnet.sigmoid(Z)
        return (p, 1 - p)

    def _open_affine_transformation(self, z):
        assert not self.no_affine

        assert hasattr(z, 'owner')
        assert z.owner is not None
        op = z.owner.op
        assert isinstance(op, T.Elemwise)
        assert isinstance(op.scalar_op, theano.scalar.basic.Add)
        dot, b = z.owner.inputs

        assert hasattr(dot, 'owner')
        assert dot.owner is not None
        assert isinstance(dot.owner.op, T.basic.Dot)
        x, W = dot.owner.inputs

        return x, W, b

    def _cost(self, Y, Y_hat):

        z = arg_of_sigmoid(Y_hat[0])
        assert z.ndim == 2
        nll_pos = T.nnet.softplus(-z)
        nll_neg = T.nnet.softplus(-z) + z

        if self._has_binary_target:

            if self.no_affine:
                # The following code is the equivalent of accessing log_prob
                # by the indices in Y, but it is written such that the
                # computation can happen on the GPU rather than CPU.
                flat_Y_pos = Y[0].flatten()
                flat_Y_neg = Y[1].flatten()
                range_ = T.arange(Y[0].shape[0])
                if self.binary_target_dim > 1:
                    # because of an error in optimization (local_useless_tile)
                    # when tiling with (1, 1)
                    range_ = T.tile(range_.dimshuffle(0, 'x'),
                                    (1, self.binary_target_dim)).flatten()
                flat_indices_pos = flat_Y_pos + range_ * self.n_classes
                flat_indices_neg = flat_Y_neg + range_ * self.n_classes
                nll_pos = nll_pos.flatten()[flat_indices_pos] \
                                .reshape(Y.shape, ndim=2)
                nll_neg = nll_neg.flatten()[flat_indices_neg] \
                                .reshape(Y.shape, ndim=2)
                nll_of = (nll_pos, nll_neg)

            else:
                # Ignore the net input and activation, access the weights and
                # biases corresponding to specified targets to get the best
                # performance
                x, W, b = self._open_affine_transformation(z)
                flat_Y_pos = Y[0].flatten()
                flat_Y_neg = Y[1].flatten()
                batch_size = Y[0].shape[0]
                W_pos = W[:, flat_Y_pos].reshape((batch_size, self.input_dim,
                                                  self.binary_target_dim))
                W_neg = W[:, flat_Y_neg].reshape((batch_size, self.input_dim,
                                                  self.binary_target_dim))
                b_pos = b[:, flat_Y_pos].reshape((batch_size,
                                                  self.binary_target_dim))
                b_neg = b[:, flat_Y_neg].reshape((batch_size,
                                                  self.binary_target_dim))
                z_pos = T.batched_dot(x, W_pos) + b_pos
                z_neg = T.batched_dot(x, W_neg) + b_neg
                nll_pos = T.nnet.softplus(-z_pos)
                nll_neg = T.nnet.softplus(-z_neg) + z_neg
                nll_of = (nll_pos, nll_neg)

        else:
            nll_of = (Y[0] * nll_pos, Y[1] * nll_neg)

        return nll_of

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        nll_pos, nll_neg = self._cost(Y, Y_hat)
        nlls = (nll_pos + nll_neg).sum(axis=1)
        assert nlls.ndim == 1

        return nlls.mean()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        rval = OrderedDict()

        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)

            if (targets is not None):
                if (self.binary_target_dim == 1):
                    # if binary_target_dim>1, the misclass rate is ill-defined
                    y_hat = T.argmax(state, axis=1)
                    y = (targets.reshape(y_hat.shape)
                         if self._has_binary_target
                         else T.argmax(targets, axis=1))
                    misclass = T.neq(y, y_hat).mean()
                    misclass = T.cast(misclass, config.floatX)
                    rval['misclass'] = misclass
                rval['nll'] = self.cost(Y_hat=state, Y=targets)

        return rval

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.np_format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt,
                                         self.input_space.axes,
                                         ('b', 0, 1, 'c'))
        return rval

    @wraps(Layer.get_weights)
    def get_weights(self):

        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        self.W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()
