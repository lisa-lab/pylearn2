from nose.tools import raises
import numpy
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.config import yaml_parse
from pylearn2.models.mlp import (
    MLP, Linear, CompositeLayer, ConvRectifiedLinear, SpaceConverter
)
from pylearn2.models.vae import VAE
from pylearn2.models.vae.kl import DiagonalGaussianPriorPosteriorKL
from pylearn2.models.vae.prior import Prior, DiagonalGaussianPrior
from pylearn2.models.vae.conditional import (
    Conditional,
    BernoulliVector,
    DiagonalGaussian
)
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import as_floatX
from pylearn2.utils import testing


class DummyVAE(object):
    rng = make_np_rng(default_seed=11223)
    batch_size = 100


class DummyPrior(Prior):
    def initialize_parameters(self, *args, **kwargs):
        self._params = []


class DummyConditional(Conditional):
    def _get_default_output_layer(self):
        return CompositeLayer(layer_name='composite',
                              layers=[Linear(layer_name='1', dim=self.ndim,
                                             irange=0.01),
                                      Linear(layer_name='2', dim=self.ndim,
                                             irange=0.01)])

    def _get_required_mlp_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.ndim),
                               VectorSpace(dim=self.ndim)])


###############################################################################
# models/vae/prior.py tests
###############################################################################
# -------------------------------- Prior --------------------------------------
def test_prior_set_vae():
    """
    Prior.set_vae adds a reference to the vae and adopts the vae's rng
    and batch_size attributes
    """
    prior = DummyPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    testing.assert_same_object(prior.vae, vae)
    testing.assert_same_object(prior.rng, vae.rng)
    testing.assert_equal(prior.batch_size, vae.batch_size)


@raises(RuntimeError)
def test_prior_raises_exception_if_called_twice():
    """
    Prior.set_vae raises an exception if it has already been called
    """
    prior = DummyPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    prior.set_vae(vae)


def test_prior_get_vae():
    """
    Prior.get_vae returns its VAE
    """
    prior = DummyPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    testing.assert_same_object(prior.get_vae(), vae)


# ------------------------- DiagonalGaussianPrior -----------------------------
def test_diagonal_gaussian_prior_initialize_parameters():
    """
    DiagonalGaussianPrior.initialize_parameters works without crashing
    """
    prior = DiagonalGaussianPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    prior.initialize_parameters(nhid=5)


def test_diagonal_gaussian_prior_sample_from_p_z():
    """
    DiagonalGaussianPrior.sample_from_p_z works without crashing
    """
    prior = DiagonalGaussianPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    prior.initialize_parameters(nhid=5)
    prior.sample_from_p_z(10)


def test_diagonal_gaussian_prior_log_p_z():
    """
    DiagonalGaussianPrior.log_p_z works without crashing
    """
    prior = DiagonalGaussianPrior()
    vae = DummyVAE()
    prior.set_vae(vae)
    prior.initialize_parameters(nhid=5)
    z = T.tensor3('z')
    prior.log_p_z(z)


###############################################################################
# models/vae/conditional.py tests
###############################################################################
# ----------------------------- Conditional -----------------------------------
@raises(ValueError)
def test_conditional_requires_nested_mlp():
    """
    Conditional rejects non-nested MLPs
    """
    mlp = MLP(nvis=10, layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    Conditional(mlp=mlp, name='conditional')


@raises(ValueError)
def test_conditional_rejects_invalid_output_layer():
    """
    Conditional rejects invalid user-defined output layer
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01),
                      Linear(layer_name='mu', dim=5, irange=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional',
                                   output_layer_required=False)
    vae = DummyVAE()
    conditional.set_vae(vae)
    conditional.initialize_parameters(input_space=VectorSpace(dim=5), ndim=5)


def test_conditional_returns_mlp_weights():
    """
    Conditional.get_weights calls its MLP's get_weights method
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    conditional.initialize_parameters(input_space=VectorSpace(dim=5), ndim=5)
    numpy.testing.assert_equal(conditional.get_weights(), mlp.get_weights())


def test_conditional_returns_lr_scalers():
    """
    Conditional.get_lr_scalers calls its MLP's get_lr_scalers method
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             W_lr_scale=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    conditional.initialize_parameters(input_space=VectorSpace(dim=5), ndim=5)
    testing.assert_equal(conditional.get_lr_scalers(), mlp.get_lr_scalers())


def test_conditional_modify_updates():
    """
    Conditional.modify_updates calls its MLP's modify_updates method
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    conditional.initialize_parameters(input_space=VectorSpace(dim=5), ndim=5)
    updates = OrderedDict(zip(mlp.get_params(), mlp.get_params()))
    testing.assert_equal(conditional.modify_updates(updates),
                         mlp.modify_updates(updates))


def test_conditional_set_vae():
    """
    Conditional.set_vae adds a reference to the vae and adopts the vae's rng
    and batch_size attributes
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    testing.assert_same_object(conditional.vae, vae)
    testing.assert_same_object(conditional.rng, vae.rng)
    testing.assert_equal(conditional.batch_size, vae.batch_size)


@raises(RuntimeError)
def test_conditional_raises_exception_if_called_twice():
    """
    Conditional.set_vae raises an exception if it has already been called
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    conditional.set_vae(vae)


def test_conditional_get_vae():
    """
    Conditional.get_vae returns its VAE
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    testing.assert_same_object(conditional.get_vae(), vae)


def test_conditional_initialize_parameters():
    """
    Conditional.initialize_parameters does the following:
    * Set its input_space and ndim attributes
    * Calls its MLP's set_mlp method
    * Sets its MLP's input_space
    * Validates its MLP
    * Sets its params and param names
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)

    testing.assert_same_object(input_space, conditional.input_space)
    testing.assert_equal(conditional.ndim, 5)
    testing.assert_same_object(mlp.get_mlp(), conditional)
    testing.assert_same_object(mlp.input_space, input_space)
    mlp_params = mlp.get_params()
    conditional_params = conditional.get_params()
    assert all([mp in conditional_params for mp in mlp_params])
    assert all([cp in mlp_params for cp in conditional_params])


def test_conditional_encode_conditional_parameters():
    """
    Conditional.encode_conditional_parameters calls its MLP's fprop method
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DummyConditional(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)

    X = T.matrix('X')
    mlp_Y1, mlp_Y2 = mlp.fprop(X)
    cond_Y1, cond_Y2 = conditional.encode_conditional_params(X)
    f = theano.function([X], [mlp_Y1, mlp_Y2, cond_Y1, cond_Y2])
    rval = f(as_floatX(numpy.random.uniform(size=(10, 5))))
    numpy.testing.assert_allclose(rval[0], rval[2])
    numpy.testing.assert_allclose(rval[1], rval[3])


# ----------------------------- BernoulliVector -------------------------------
def test_bernoulli_vector_default_output_layer():
    """
    BernoulliVector's default output layer is compatible with its required
    output space
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = BernoulliVector(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)


def test_bernoulli_vector_sample_from_conditional():
    """
    BernoulliVector.sample_from_conditional works when num_samples is provided
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = BernoulliVector(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    conditional.sample_from_conditional([mu], num_samples=2)


@raises(ValueError)
def test_bernoulli_vector_reparametrization_trick():
    """
    BernoulliVector.sample_from_conditional raises an error when asked to
    sample using the reparametrization trick
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = BernoulliVector(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    epsilon = T.tensor3('epsilon')
    conditional.sample_from_conditional([mu], epsilon=epsilon)


def test_bernoulli_vector_conditional_expectation():
    """
    BernoulliVector.conditional_expectation doesn't crash
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = BernoulliVector(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    conditional.conditional_expectation([mu])


def test_bernoulli_vector_log_conditional():
    """
    BernoulliVector.log_conditional doesn't crash
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = BernoulliVector(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    samples = T.tensor3('samples')
    conditional.log_conditional(samples, [mu])


# ---------------------------- DiagonalGaussian -------------------------------
def test_diagonal_gaussian_default_output_layer():
    """
    DiagonalGaussian's default output layer is compatible with its required
    output space
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)


def test_diagonal_gaussian_sample_from_conditional():
    """
    DiagonalGaussian.sample_from_conditional works when num_samples is provided
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    log_sigma = T.matrix('log_sigma')
    conditional.sample_from_conditional([mu, log_sigma], num_samples=2)


def test_diagonal_gaussian_reparametrization_trick():
    """
    DiagonalGaussian.sample_from_conditional works when asked to sample using
    the reparametrization trick
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    log_sigma = T.matrix('log_sigma')
    epsilon = T.tensor3('epsilon')
    conditional.sample_from_conditional([mu, log_sigma], epsilon=epsilon)


def test_diagonal_gaussian_conditional_expectation():
    """
    DiagonalGaussian.conditional_expectation doesn't crash
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    log_sigma = T.matrix('log_sigma')
    conditional.conditional_expectation([mu, log_sigma])


def test_diagonal_gaussian_log_conditional():
    """
    DiagonalGaussian.log_conditional doesn't crash
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    mu = T.matrix('mu')
    log_sigma = T.matrix('log_sigma')
    samples = T.tensor3('samples')
    conditional.log_conditional(samples, [mu, log_sigma])


def test_diagonal_gaussian_sample_from_epsilon():
    """
    DiagonalGaussian.sample_from_epsilon doesn't crash
    """
    mlp = MLP(layers=[Linear(layer_name='h', dim=5, irange=0.01,
                             max_col_norm=0.01)])
    conditional = DiagonalGaussian(mlp=mlp, name='conditional')
    vae = DummyVAE()
    conditional.set_vae(vae)
    input_space = VectorSpace(dim=5)
    conditional.initialize_parameters(input_space=input_space, ndim=5)
    conditional.sample_from_epsilon((2, 10, 5))


###############################################################################
# models/vae/__init__.py tests
###############################################################################
def test_one_sample_allowed():
    """
    VAE allows one sample per data point
    """
    encoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = BernoulliVector(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X, num_samples=1)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(as_floatX(rng.uniform(size=(10, 10))))


def test_multiple_samples_allowed():
    """
    VAE allows multiple samples per data point
    """
    encoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = BernoulliVector(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X, num_samples=10)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(as_floatX(rng.uniform(size=(10, 10))))


def test_convolutional_compatible():
    """
    VAE allows convolutional encoding networks
    """
    encoding_model = MLP(
        layers=[
            SpaceConverter(
                layer_name='conv2d_converter',
                output_space=Conv2DSpace(shape=[4, 4], num_channels=1)
            ),
            ConvRectifiedLinear(
                layer_name='h',
                output_channels=2,
                kernel_shape=[2, 2],
                kernel_stride=[1, 1],
                pool_shape=[1, 1],
                pool_stride=[1, 1],
                pool_type='max',
                irange=0.01)
            ]
    )
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=16, irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = BernoulliVector(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=16, prior=prior, conditional=conditional,
              posterior=posterior, nhid=16)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X, num_samples=10)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(as_floatX(rng.uniform(size=(10, 16))))


def test_vae_automatically_finds_kl_integrator():
    """
    VAE automatically finds the right KLIntegrator
    """
    encoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = BernoulliVector(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)
    assert (vae.kl_integrator is not None and
            isinstance(vae.kl_integrator, DiagonalGaussianPriorPosteriorKL))


###############################################################################
# costs/vae.py tests
###############################################################################
def test_VAE_cost():
    """
    VAE trains properly with the VAE cost
    """
    train_object = yaml_parse.load_path('pylearn2/models/tests/'
                                        'test_vae_cost_vae_criterion.yaml')
    train_object.main_loop()


def test_IS_cost():
    """
    VAE trains properly with the importance sampling cost
    """
    train_object = yaml_parse.load_path('pylearn2/models/tests/'
                                        'test_vae_cost_is_criterion.yaml')
    train_object.main_loop()
