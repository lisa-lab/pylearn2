from nose.tools import raises
import theano
import theano.tensor as T
from pylearn2.config import yaml_parse
from pylearn2.models.mlp import (
    MLP, Linear, CompositeLayer, ConvRectifiedLinear, SpaceConverter
)
from pylearn2.models.vae import VAE
from pylearn2.models.vae.kl import DiagonalGaussianPriorPosteriorKL
from pylearn2.models.vae.prior import DiagonalGaussianPrior
from pylearn2.models.vae.conditional import BernoulliVector, DiagonalGaussian
from pylearn2.space import Conv2DSpace
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import as_floatX


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


def test_diagonal_gaussian_prior_diagonal_gaussian_posterior():
    """
    DiagonalGaussianPrior and DiagonalGaussian works without crashing
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


def test_bernoulli_vector_conditional():
    """
    BernoulliVector works without crashing
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


def test_diagonal_gaussian_conditional():
    """
    DiagonalGaussian works without crashing
    """
    encoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = DiagonalGaussian(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X, num_samples=1)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(as_floatX(rng.uniform(size=(10, 10))))


def test_output_layer_not_required():
    """
    Conditional allows user-defined output layers in MLP
    """
    encoding_model = MLP(
        layers=[
            Linear(layer_name='h', dim=10, irange=0.01),
            CompositeLayer(
                layer_name='phi',
                layers=[
                    Linear(layer_name='mu', dim=5, irange=0.01),
                    Linear(layer_name='log_sigma', dim=5, irange=0.01)
                ]
            )
        ]
    )
    decoding_model = MLP(
        layers=[
            Linear(layer_name='h', dim=10, irange=0.01),
            CompositeLayer(
                layer_name='theta',
                layers=[
                    Linear(layer_name='mu', dim=10, irange=0.01),
                    Linear(layer_name='log_sigma', dim=10, irange=0.01)
                ]
            )
        ]
    )
    prior = DiagonalGaussianPrior()
    conditional = DiagonalGaussian(mlp=decoding_model,
                                   name='conditional',
                                   output_layer_required=False)
    posterior = DiagonalGaussian(mlp=encoding_model,
                                 name='posterior',
                                 output_layer_required=False)
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)


@raises(ValueError)
def test_conditional_rejects_invalid_output_layer():
    """
    Conditional rejects invalid user-defined output layer
    """
    encoding_model = MLP(
        layers=[
            Linear(layer_name='h', dim=10, irange=0.01),
            CompositeLayer(
                layer_name='phi',
                layers=[
                    Linear(layer_name='mu', dim=5, irange=0.01),
                    Linear(layer_name='log_sigma', dim=5, irange=0.01)
                ]
            )
        ]
    )
    decoding_model = MLP(
        layers=[
            Linear(layer_name='h', dim=10, irange=0.01),
            CompositeLayer(
                layer_name='theta',
                layers=[
                    Linear(layer_name='mu', dim=8, irange=0.01),
                    Linear(layer_name='log_sigma', dim=8, irange=0.01)
                ]
            )
        ]
    )
    prior = DiagonalGaussianPrior()
    conditional = BernoulliVector(mlp=decoding_model,
                                  name='conditional',
                                  output_layer_required=False)
    posterior = DiagonalGaussian(mlp=encoding_model,
                                 name='posterior',
                                 output_layer_required=False)
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)


@raises(ValueError)
def test_conditional_requires_nested_mlp():
    """
    Visible rejects non-nested MLPs
    """
    decoding_model = MLP(nvis=10,
                         layers=[Linear(layer_name='h', dim=10, irange=0.01)])
    conditional = BernoulliVector(mlp=decoding_model, name='conditional')


def test_lr_scalers_returned():
    """
    VAE return its encoding and decoding models' LR scalers
    """
    encoding_model = MLP(layers=[Linear(layer_name='h', dim=10, W_lr_scale=0.5,
                                        irange=0.01)])
    decoding_model = MLP(layers=[Linear(layer_name='h', dim=10, W_lr_scale=0.5,
                                        irange=0.01)])
    prior = DiagonalGaussianPrior()
    conditional = DiagonalGaussian(mlp=decoding_model, name='conditional')
    posterior = DiagonalGaussian(mlp=encoding_model, name='posterior')
    vae = VAE(nvis=10, prior=prior, conditional=conditional,
              posterior=posterior, nhid=5)
    lr_scalers = vae.get_lr_scalers()
    expected_keys = [encoding_model.layers[0].transformer.get_params()[0],
                     decoding_model.layers[0].transformer.get_params()[0]]
    assert all(key in lr_scalers.keys() for key in expected_keys)
    assert all(key in expected_keys for key in lr_scalers.keys())
    assert all(lr_scalers[key] == 0.5 for key in expected_keys)


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


def test_VAE_cost():
    """
    VAE trains properly with the VAE cost
    """
    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_dense_design_matrix {
            rng: !obj:pylearn2.utils.rng.make_np_rng {
                default_seed: 11234,
            },
            num_examples: 100,
            dim: &nvis 10,
            num_classes: 2,
        },
        model: !obj:pylearn2.models.vae.VAE {
            nvis: *nvis,
            nhid: &nhid 5,
            prior: !obj:pylearn2.models.vae.prior.DiagonalGaussianPrior {},
            conditional: !obj:pylearn2.models.vae.conditional.BernoulliVector {
                name: 'conditional',
                mlp: !obj:pylearn2.models.mlp.MLP {
                    layers: [
                        !obj:pylearn2.models.mlp.Linear {
                            layer_name: 'h_d',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
            posterior: !obj:pylearn2.models.vae.conditional.DiagonalGaussian {
                name: 'posterior',
                mlp: !obj:pylearn2.models.mlp.MLP {
                    layers: [
                        !obj:pylearn2.models.mlp.RectifiedLinear {
                            layer_name: 'h_e',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 100,
            learning_rate: 1e-3,
            monitoring_dataset: {
                'train' : *train,
            },
            cost: !obj:pylearn2.costs.vae.VAECriterion {
                num_samples: 2,
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 2
            },
        },
    }
    """
    train_object = yaml_parse.load(yaml_string)
    train_object.main_loop()


def test_IS_cost():
    """
    VAE trains properly with the importance sampling cost
    """
    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_dense_design_matrix {
            rng: !obj:pylearn2.utils.rng.make_np_rng {
                default_seed: 11234,
            },
            num_examples: 100,
            dim: &nvis 10,
            num_classes: 2,
        },
        model: !obj:pylearn2.models.vae.VAE {
            nvis: *nvis,
            nhid: &nhid 5,
            prior: !obj:pylearn2.models.vae.prior.DiagonalGaussianPrior {},
            conditional: !obj:pylearn2.models.vae.conditional.BernoulliVector {
                name: 'conditional',
                mlp: !obj:pylearn2.models.mlp.MLP {
                    layers: [
                        !obj:pylearn2.models.mlp.Linear {
                            layer_name: 'h_d',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
            posterior: !obj:pylearn2.models.vae.conditional.DiagonalGaussian {
                name: 'posterior',
                mlp: !obj:pylearn2.models.mlp.MLP {
                    layers: [
                        !obj:pylearn2.models.mlp.RectifiedLinear {
                            layer_name: 'h_e',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 100,
            learning_rate: 1e-3,
            monitoring_dataset: {
                'train' : *train,
            },
            cost: !obj:pylearn2.costs.vae.ImportanceSamplingCriterion {
                num_samples: 2,
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 2
            },
        },
    }
    """
    train_object = yaml_parse.load(yaml_string)
    train_object.main_loop()
