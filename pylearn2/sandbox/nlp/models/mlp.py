"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import theano.tensor as T
from theano import config

from pylearn2.models import mlp
from pylearn2.models.mlp import Layer
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.compat import OrderedDict


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

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

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
