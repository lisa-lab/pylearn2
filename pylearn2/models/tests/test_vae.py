import theano
import theano.tensor as T
from pylearn2.config import yaml_parse
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Linear, ConvRectifiedLinear
from pylearn2.models.vae import VAE
from pylearn2.models.vae.visible import BinaryVisible
from pylearn2.models.vae.latent import DiagonalGaussianPrior
from pylearn2.space import Conv2DSpace
from pylearn2.utils.rng import make_np_rng


def test_one_sample_allowed():
    """
    VAE allows one sample per data point
    """
    encoding_model = MLP(nvis=10, layers=[Linear(layer_name='h', dim=10,
                                                 irange=0.01)])
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=10,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=1)
    vae = VAE(nvis=10, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(rng.uniform(size=(10, 10)))


def test_multiple_samples_allowed():
    """
    VAE allows multiple samples per data point
    """
    encoding_model = MLP(nvis=10, layers=[Linear(layer_name='h', dim=10,
                                                 irange=0.01)])
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=10,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=10)
    vae = VAE(nvis=10, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(rng.uniform(size=(10, 10)))


def test_convolutional_compatible():
    """
    VAE allows convolutional encoding networks
    """
    encoding_model = MLP(
        input_space=Conv2DSpace(shape=[4, 4], num_channels=1),
        layers=[ConvRectifiedLinear(
            layer_name='h',
            output_channels=2,
            kernel_shape=[2, 2],
            kernel_stride=[1, 1],
            pool_shape=[1, 1],
            pool_stride=[1, 1],
            pool_type='max',
            irange=0.01
        )]
    )
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=16,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=10)
    vae = VAE(nvis=16, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    rng = make_np_rng(default_seed=11223)
    f(rng.uniform(size=(10, 16)))


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
            visible: !obj:pylearn2.models.vae.visible.BinaryVisible {
                decoding_model: !obj:pylearn2.models.mlp.MLP {
                    nvis: *nhid,
                    layers: [
                        !obj:pylearn2.models.mlp.Linear {
                            layer_name: 'h_d',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
            latent: !obj:pylearn2.models.vae.latent.DiagonalGaussianPrior {
                encoding_model: !obj:pylearn2.models.mlp.MLP {
                    nvis: *nvis,
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
            cost: !obj:pylearn2.costs.vae.VAECriterion {},
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
            visible: !obj:pylearn2.models.vae.visible.BinaryVisible {
                decoding_model: !obj:pylearn2.models.mlp.MLP {
                    nvis: *nhid,
                    layers: [
                        !obj:pylearn2.models.mlp.Linear {
                            layer_name: 'h_d',
                            dim: 10,
                            irange: .01,
                        },
                    ],
                },
            },
            latent: !obj:pylearn2.models.vae.latent.DiagonalGaussianPrior {
                encoding_model: !obj:pylearn2.models.mlp.MLP {
                    nvis: *nvis,
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
